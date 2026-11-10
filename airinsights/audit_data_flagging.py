#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:15:20 2026

"""
"""
@author: TD Enviro
"""


import pandas as pd
import numpy as np
import re
import os
import plotly.graph_objects as go
import sys
from pathlib import Path
from airinsights.helpers import _validate_hourly
from airinsights.audit_helpers import default_range_thresholds, default_roc_thresholds, default_sticking_hrs_thresholds, common_error_codes, unrealistic_pollutant_comparisons



def audit_data_flagging(input_data : pd.DataFrame,
                    config_dict : dict,
                    range_thresholds : dict | None = {}, 
                    roc_thresholds : dict | None = {},
                    sticking_hrs_thresholds : dict | None = {},
                    remove_invalid: bool = True,
                    remove_suspect: bool = False,
                    output_file_path : str | None = os.getcwd()
                    ):
    """This function flags data as suspect or invalid based on pollutant-specific thresholds for range checks, rate of change (rolling 2hr centered), 
        sticking data, and unrealistic pollutant relationships. User has option to flag and remove data or just flag.
            Unrealistic pollutant relationships:    
                O3: high value (70 ppb) or absolute rate of change (40 ppb) in overnight and early morning hours (11PM - 6AM Local Time)
                PM: PM2_5>PM10
                NO,NO2: NO>NOx or NO2>NOx
    
    Parameters
    ----------
    input_data : pd.DataFrame
        A pandas DataFrame containing hourly AQ data (read in using read_aqdata_file)
    
    config_dict : dict
        A dictionary detailing the input_data structure, containing input parameter names and values. 
    
    range_thresholds : dict, optional
        min,max ranges per pollutant to use for range check. Default or user-defined. {str:{'min':int or float,'min':int or float}}
        If only some pollutants specified in user-defined thresholds, program will try to fill in defaults.
        Default range thresholds - {'BC':{'min':-400,'max':500000},'CO':{'min':0,'max':15},'NO2':{'min':-1,'max':500},
                                          'NO':{'min':-1,'max':3000},'O3':{'min':-5,'max':250},'PM2_5':{'min':-5,'max':1000},
                                          'PM10':{'min':-5,'max':10000},'SO2':{'min':-5,'max':300},'VOC':{'min':-1,'max':300}}
        Default range threshold units - (BC ng/m^3, CO ppm, NO2 ppb, NO ppb, O3 ppb, PM2_5 ug/m^3, PM10 ug/m^3,SO2 ppb,VOC ppm)
    
    roc_thresholds : dict, optional
        Absolute rate of change thresholds per pollutant to use for rate of change check. Default or user-defined. {str:int or float}
        If only some pollutants specified in user-defined thresholds, program will try to fill in defaults.
        Default rate of change thresholds - {'NO2':50,'O3':60,'PM2_5':150,'PM10':300}
        Default rate of change threshold units - (NO2 ppb, O3 ppb, PM2_5 ug/m^3, PM10 ug/m^3)
    
    sticking_hrs_thresholds : dict, optional
        # of sticking hours and associated minimum value  per pollutant for sticking hour check. Default or user-defined. {str:{'stck_hr_count':int,'min_val':int or float}}
        If only some pollutants specified in user-defined thresholds, program will try to fill in defaults.
        Default sticking hrs thresholds - {'NO2':{'stck_hr_count':10,'min_val':5},'O3':{'stck_hr_count':12,'min_val':10},
                                                 'PM2_5':{'stck_hr_count':6,'min_val':5},'PM10':{'stck_hr_count':6,'min_val':10}}
        Default sticking hrs thresholds units - (NO2 ppb, O3 ppb, PM2_5 ug/m^3, PM10 ug/m^3)
    
    remove_invalid : bool, default True
        Invalid data is automatically removed from DataFrame. If False, invalid data is kept in DataFrame.
    
    remove_suspect : bool, default False
        Suspect data is initally kept in the DataFrame, prompting user review and decision (to keep or remove). If True, suspect data is automatically removed (set to nan) from DataFrame.
    
    output_file_path : str, optional
        The file path where output .txt files and figures should be saved. If None provided, current working directory will be used.
    
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing quality-controlled data with all input data columns and the following appended columns:
                
            **QC_Flag**: quality control flags - Invalid(I) and Suspect(S) 
            
            **Flagging_Reason**: the check(s) or reason(s) that resulted in a QC_Flag

    Data Flagging Thresholds (.csv saved to output_file_path)
        A pandas DataFrame describing the flagging checks performed per pollutant and their respective thresholds and associated QC_Flag. Contains the following columns:
            
            **pollutant_col**: pollutant in input_data (column name provided by user in config_dict)
           
            **Flagging_Check**: name of flagging check (e.g., range)
            
            **QC_Flag**: data action associated from flagging check (I or S)
            
            **Threshold(s)**: thresholds used for specific flagging_check for associated pollutant 
            
        
    Notes
    -----
    
    If ``remove_suspect`` is ``False`` and Suspect data is found, 
        also outputs interative plotly time series of Suspect (and Invalid) data for review (.html's saved to new "Suspect Data for Review" folder in output_file_path).
    
    If Suspect or Invalid data makes up 1% or more of all input data for a certain site and pollutant pair, 
        also outputs Data Flagging Highlights (.csv saved to output_file_path), highlighting site,pollutant pairs with 1% or more of data.
        
        **site_col**: site of site,pollutant pair that was flagged (column name provided by user in config_dict)
        
        **pollutant_col**: pollutant of site,pollutant pair that was flagged (column name provided by user in config_dict)
        
        **QC_Flag**: quality control flag that makes up 1% or more of data  - Invalid(I) and Suspect(S) 
        
        **Flag_Percent**: percentage of all input data that was flagged with this QC_Flag
        
        **Flagging_Reasons**: list of unique Flagging Reasons that resulted in the associated QC_Flag


    """



    
    
    
    
    # Make a copy of the user input data and check if hourly (will raise an error if data only contains non-hourly freq or will print out how much data needed to be excluded)
    df = input_data.copy()
    df = _validate_hourly(df,config_dict)
    
    # Inverse pollutant map (to use for data mapping)
    inv_pollutant_map = {v: k for k, v in config_dict['pollutant_map'].items()}
    
    # Add Standard Pollutant column to dataframe
    df['Standard_Pollutant'] = df[config_dict['pollutant_col']].map(inv_pollutant_map)
    
    # Subset config_dict to pollutants that are actually in the dataset
    p_in_df = df['Standard_Pollutant'].unique()
    config_dict['pollutant_map'] = {k: config_dict['pollutant_map'][k] for k in p_in_df}
    

    # Define generic thresholds for ROC and sticking to reference in dictionary merges below
    #TODO - Discuss - setting generic ROC threshold based on unQC'ed data - could move this after range check so that invalid high/low data doesn't bias this?
    generic_roc = (round((df.groupby('Standard_Pollutant')[config_dict['value_col']].mean() * 10),3).to_dict())
    
    generic_sticking = {p : default_sticking_hrs_thresholds.get('Generic') for p in p_in_df}
    
        
    #--Set default or generic thresholds where not defined (for duplicate keys, righthand dict values trump lefthand side in each union pair)
    
    range_thresholds = default_range_thresholds | range_thresholds
    
    roc_thresholds = generic_roc | (default_roc_thresholds | roc_thresholds)
    
    sticking_hrs_thresholds = generic_sticking | (default_sticking_hrs_thresholds | sticking_hrs_thresholds)
    

    #-- Check broad format of thresholds and other input fields

    # Range thresholds
    if not isinstance(range_thresholds, dict): 
        raise TypeError(f"dictionary expected for range_thresholds, got {type(range_thresholds).__name__}")
     
    # ROC thresholds   
    if not isinstance(roc_thresholds, dict): 
        raise TypeError(f"dictionary expected for roc_thresholds, got {type(roc_thresholds).__name__}")
    
     
    # Sticking thresholds
    if not isinstance(sticking_hrs_thresholds, dict): 
        raise TypeError(f"dictionary expected for sticking_hrs_thresholds, got {type(sticking_hrs_thresholds).__name__}")
    
    # Output_file_path
    if not isinstance(output_file_path, str): 
        raise TypeError(f"string expected for output_file_path, got {type(output_file_path).__name__}")
     
    #-- Check contents of thresholds and populate in threshold table
    threshold_tbl = []
    # Range thresholds
    for k,v in range_thresholds.items():
    
        if not isinstance(v, dict): 
            raise TypeError(f"Dictionary with range min and max expected, got {type(v).__name__} for {k}.")
        
        elif 'min' not in v.keys() or 'max' not in v.keys():
            raise KeyError(f"Missing keys for {k} range threshold. Both min and max keys needed per pollutant.")
        
        # Check that min,max in p_range provided correctely (max>min)
        elif v['min'] >= v['max']:
            raise ValueError(f"Issue with user-provided range threshold for {k}. Min is greater than or equal to max.")
            
        elif k in p_in_df:
            threshold_tbl.append({'Standard_Pollutant':k,'Flagging_Check':'Range','QC_Flag':'I','Threshold(s)':v})
    
    # ROC thresholds 
    for k,v in roc_thresholds.items():       
        
        if not isinstance(v, int) and (not isinstance(v,float)): 
            raise TypeError(f"Integer or float expected for rate of change, got {type(v).__name__} for {k}.")
            
        elif v<0:
            raise ValueError(f"Absolute rate of change value cannot be negative. Fix threshold for {k}.")
            
        elif k in p_in_df:
            threshold_tbl.append({'Standard_Pollutant':k,'Flagging_Check':'ROC','QC_Flag':'S','Threshold(s)':v})

    
    # Sticking thresholds    
    for k,v in sticking_hrs_thresholds.items():
        
        if not isinstance(v, dict): 
            raise TypeError(f"Dictionary with stck_hr_count and min_val expected in sticking_hr_thresholds, got {type(v).__name__} for {k}.")
            
        elif 'stck_hr_count' not in v.keys() or ('min_val' not in v.keys()):
            raise KeyError(f"Missing keys for {k} sticking_hr_thresholds. Both stck_hr_count and min_val keys needed per pollutant.")
        
        elif k in p_in_df:
            threshold_tbl.append({'Standard_Pollutant':k,'Flagging_Check':'Sticking','QC_Flag':'S','Threshold(s)':v})
    
    # Add common error codes to threshold tbl for every pollutant
    for p in p_in_df:
        threshold_tbl.append({'Standard_Pollutant':p,'Flagging_Check':'Common Error Codes','QC_Flag':'S','Threshold(s)':common_error_codes})


    # Print status statement
    print('Data are undergoing invalid flagging review....',end='\n\n')
    
    #-- Add flagging-related columns and create group_cols (used for ROC calculations) and full_count (used later for % of flagged data)
    
    # Organize df
    df = df.sort_values(by=[config_dict['site_col'],config_dict['pollutant_col'],config_dict['timestamp_col']]).reset_index(drop=True)
    
    # Add flagging-related columns to add to over coarse of data flagging
    df['QC_Flag'] = '' #Create new column in df for QC Flags (initially blank)
    df['Flagging_Reason'] = '' # Create new column in df for Suspect Reasons (initially blank)

    # Count number of non-NaN entries per site,pollutant pair before any QC (used later to calculate % of flagged data) 
    group_cols = [c for c in df.columns if c not in [config_dict['value_col'], config_dict['timestamp_col'],'QC_Flag','Flagging_Reason']]
    full_count = df.groupby([config_dict['site_col'],config_dict['pollutant_col']])[config_dict['value_col']].count().reset_index().rename(columns = {config_dict['value_col'] :'Full_Count'})
    

    #-- Perform all INVALID data checks
    
    #-- Perform Range Check
    df[['Range_Min','Range_Max']] = pd.json_normalize(df['Standard_Pollutant'].map(range_thresholds))
    df.loc[
        ((df[config_dict['value_col']]<df['Range_Min']) | (df[config_dict['value_col']]>df['Range_Max'])),
        ['QC_Flag', 'Flagging_Reason']] = ['I','RG']

    # Keep dataframe with all data (for plotting later)
    full_df = df.copy()
    
    # By default, remove (nan) invalid data. If False, keep.
    if remove_invalid:
        
        print('Invalid data removed based on user input.',end='\n\n')
        
        df.loc[df['QC_Flag']=='I',config_dict['value_col']] = np.nan
    
    
    #-- Perform all SUSPECT data checks (note: specifically doesn't review data already flagged as Invalid, this may be retained depending on remove_invalid bool)
    
    print('Data are undergoing suspect flagging review...',end='\n\n')
    
    #-- Calculate rate of change (ROC) and rolling ROC (after invalid data checks) to use across suspect checks
    df['ROC'] = abs(df.groupby(group_cols)[config_dict['value_col']].diff().reset_index(drop=True))    
    df['ROC_roll'] = df.groupby(group_cols)['ROC'].transform(lambda r: r.rolling(window=2, min_periods=1).mean().shift(-1))
    
    
    #-- Perform common error code checks on all data (regardless of pollutant) 
    
    # Review data and add qc flag and flagging reason
    common_error_code_check = df[config_dict['value_col']].isin(common_error_codes) & (df['QC_Flag'] != 'I')
    df.loc[common_error_code_check,['QC_Flag', 'Flagging_Reason']] = ['S','CE'] #Ok to use .loc here, since first time adding suspect flagging reason

    #-- Perform Rate of Change check
    
    # Apply roc_thresholds to data
    df['ROC_Check'] = df['Standard_Pollutant'].map(roc_thresholds)
    
    # Review data and add qc flag and flagging reason
    roc_check = ((df['ROC_roll']>df['ROC_Check']) & (df['QC_Flag'] != 'I'))
    df.loc[roc_check, ['QC_Flag']] = 'S'
    df['Flagging_Reason'] = np.where(roc_check, df['Flagging_Reason'] + 'RC', df['Flagging_Reason'])
    
    #-- Perform Sticking check
    
    # Note: don't need to group by group_cols again, since ROC is calculated based on that earlier
    df['Sticking_Count'] = df.groupby((df['ROC']!=0).cumsum()).transform('size')
    
    # Apply sticking_hrs_thresholds to data
    df[['Sticking_Count_Min','Sticking_Val_Min']] = pd.json_normalize(df['Standard_Pollutant'].map(sticking_hrs_thresholds))
    
    # Review data and add qc flag and flagging reason
    sticking_check =  (df['Sticking_Count']>=df['Sticking_Count_Min']) & (df[config_dict['value_col']]>=df['Sticking_Val_Min']) & (df['QC_Flag'] != 'I')
    df.loc[sticking_check,'QC_Flag'] = 'S'
    df['Flagging_Reason'] = np.where(sticking_check, df['Flagging_Reason'] + 'ST', df['Flagging_Reason'])
    
    # Drop temporary columns from suspect checks above
    df.drop(['Range_Min','Range_Max','ROC_Check','ROC_roll','Sticking_Count','Sticking_Count_Min','Sticking_Val_Min'],axis=1,inplace=True) 
    

    #-- Perform "Unrealistic Pollution Behavior" checks for relevant pollutants
    
    #-- Perform Unrealistic Pollution Behavior check for O3 (different format than other unr checks)
    if 'O3' in config_dict['pollutant_map'].keys():
        
        # Set suspect check condition for O3
        unr_o3_condition = ((df['Standard_Pollutant']=='O3') 
                            & ((df[config_dict['timestamp_col']].dt.hour >= 23) | (df[config_dict['timestamp_col']].dt.hour <= 6)) 
                            & ((df[config_dict['value_col']]>70) | (df['ROC']>40)))
        
        # Review data and add qc flag and flagging reason
        df.loc[unr_o3_condition & (df['QC_Flag'] != 'I'),'QC_Flag'] = 'S'
        df['Flagging_Reason'] = np.where(unr_o3_condition & (df['QC_Flag'] != 'I'), df['Flagging_Reason'] + 'UO', df['Flagging_Reason'])
        
        # Add this threshold check to the threshold tbl
        threshold_tbl.append({'Standard_Pollutant':'O3','Flagging_Check':'Unrealistic Pollutant Behavior','QC_Flag':'S','Threshold(s)':'See Documentation...'})
        
    # Drop ROC column
    df = df.drop('ROC',axis=1)
    
    
    #-- Perform remaining "Unrealistic Pollution Behavior" checks - defined in unr_pollutant_comparisons (all work better with wide data format)
    for i in range(0,len(unrealistic_pollutant_comparisons['unr_ps'])):
        
        # Get pollutants per unr check and coinciding flagging reason code
        unr_lst = unrealistic_pollutant_comparisons['unr_ps'][i]
        unr_sr = unrealistic_pollutant_comparisons['flagging_reason'][i]
        
        # Only review data for unr checks if all pollutants per comparison pair are present
        if all(unr_p in config_dict['pollutant_map'].keys() for unr_p in unr_lst):
            
            # Pivoting wider to find site + timestamp pairs where certain pollutants are showing unrealistic pollutant behavior - just using df_pivot to find indexes, not melting back to df (since flagging columns are pollutant-specific and don't work in wide format)
            index_cols = [config_dict['timestamp_col'],config_dict['site_col']]
            df_pivot = df.pivot(index=index_cols,columns='Standard_Pollutant',values = config_dict['value_col']).reset_index()

            # Get site + timestamp pair where any unrealistic pollutant comparison (x>y)
            unr_sites_dates = pd.MultiIndex.from_frame(df_pivot[(df_pivot[unr_lst[0]] > df_pivot[unr_lst[1]])][[config_dict['site_col'],config_dict['timestamp_col']]])
            
            # Find rows in df where site and datetime pair equal those identified in unr_sites_dates AND pollutant column is equal to either pollutants in unr_lst
            unr_idx = pd.MultiIndex.from_frame(df[[config_dict['site_col'],config_dict['timestamp_col']]]).isin(unr_sites_dates) & df['Standard_Pollutant'].isin(unr_lst)
            
            # Add qc flag and flagging reason 
            df.loc[unr_idx & (df['QC_Flag'] != 'I'),'QC_Flag'] = 'S'  
            df['Flagging_Reason'] = np.where(unr_idx & (df['QC_Flag'] != 'I'), df['Flagging_Reason'] + unr_sr, df['Flagging_Reason'])
            
            # Add these unrealistic pollutant checks to the threshold tbl
            threshold_tbl.append({'Standard_Pollutant':unr_lst[0],'Flagging_Check':'Unrealistic Pollutant Behavior','QC_Flag':'S','Threshold(s)':f"{config_dict['pollutant_map'][unr_lst[0]]}>{config_dict['pollutant_map'][unr_lst[1]]}"})

            threshold_tbl.append({'Standard_Pollutant':unr_lst[1],'Flagging_Check':'Unrealistic Pollutant Behavior','QC_Flag':'S','Threshold(s)':f"{config_dict['pollutant_map'][unr_lst[0]]}>{config_dict['pollutant_map'][unr_lst[1]]}"})
    
    
    #-- Process Suspect Data
    
    
    #-- Remove (nan) suspect data if remove_suspect = True.
    if remove_suspect:
        
        print('Suspect data removed based on user input.',end='\n\n')
                
        df.loc[df['QC_Flag']=='S',config_dict['value_col']] = np.nan
        
        suspect_action = 'y'
    
    
    #-- Otherwise, if any Suspect data in dataframes and remove_suspect=False, create suspect data plots for user to review and ask for concurrence on removing/keeping
    #TODO - In future version, distinguish plots by Flagging Reason - so that user can review more specifically per flagging reason and concur that way
    elif 'S' in df['QC_Flag'].unique() and not remove_suspect:
        
        
        # Separate Suspect Reasons by space (every two letters - all flagging codes are two letters long)
        df['Flagging_Reason'] = df['Flagging_Reason'].apply(lambda x: ' '.join(re.findall('..',x)))
        
        
        # Merge back into full_df that has all data (none naned) but leave out df's Invalid flagged to not duplicate/overwrite with nans
        full_df = pd.concat([df[df['QC_Flag']!='I'],full_df],axis=0).drop_duplicates().sort_values(by=[config_dict['site_col'],config_dict['pollutant_col'],config_dict['timestamp_col']])

        
        suspect_sites_pollutants = full_df[full_df['QC_Flag']=='S'][[config_dict['site_col'],config_dict['pollutant_col']]].drop_duplicates().reset_index(drop=True)
        
        # Print interactive plots for all sites/pollutants with suspect data
        
        # Create subfolder in output_file_path to store suspect time series for review
        suspect_folder = Path(output_file_path,'Suspect Data for Review/')
        os.makedirs(suspect_folder,exist_ok=True)
        
        plot_count = 0
        
        print('Plotting suspect data for review...')
        
        for _,row in suspect_sites_pollutants.iterrows():
            
            sus_site = row[config_dict['site_col']]
            
            sus_p = row[config_dict['pollutant_col']]
            
            p_df = full_df[(full_df[config_dict['site_col']]==sus_site) & (full_df[config_dict['pollutant_col']]==sus_p)]
                        
            p_df_sus = p_df[p_df['QC_Flag']=='S']
            
            # Plot all data in sus_p_df, highlighting suspect data in red, and including Flagging_Reason in hover over text
            
            fig = go.Figure()
            # Have to define customdata if want to show per value Suspect Reason (x, y or customdata recognized with %). <extra></extra> removes the hover legend text outside of the box that default appears
            fig.add_trace(go.Scatter(x=list(p_df[config_dict['timestamp_col']]), y=list(p_df[config_dict['value_col']]), name='Valid',mode='lines+markers',marker_color='dimgrey',hovertemplate=("%{x}<br>" f"{sus_p}: " + "%{y}<extra></extra>")))
                     
            fig.add_trace(go.Scatter(x=list(p_df_sus[config_dict['timestamp_col']]), y=list(p_df_sus[config_dict['value_col']]), name='Suspect',mode='markers',marker_color='orange',customdata=p_df_sus['Flagging_Reason'],hovertemplate=("%{x}<br>" f"{sus_p}: " + "%{y}<br>" "Flagging Reason Code: %{customdata}<extra></extra>")))
            
            # Highlight invalid data specifically in red (helpful for reviewing suspect alongside invalid for context)
            p_df_inv = full_df[(full_df[config_dict['site_col']]==sus_site) & (full_df['QC_Flag']=='I') & (full_df[config_dict['pollutant_col']]==sus_p)]
            fig.add_trace(go.Scatter(x=list(p_df_inv[config_dict['timestamp_col']]), y=list(p_df_inv[config_dict['value_col']]), name='Invalid',mode='markers',marker_color='red',customdata=p_df_inv['Flagging_Reason'],hovertemplate=("%{x}<br>" f"{sus_p}: " + "%{y}<br>" "Flagging Reason Code: %{customdata}<extra></extra>")))
                                                                              

            title = f'Suspect {sus_p} at {sus_site}'
            n = ((p_df[config_dict['timestamp_col']].max() - p_df[config_dict['timestamp_col']].min()).days*24)/10 # number of hours to space x tick marks by
            
            fig.update_layout(title=title,
                               xaxis_title=config_dict['timestamp_col'],
                               yaxis_title=f'Hourly {sus_p}', #TODO add unit to y axis once added to config_dict?
                               legend=dict(title='Legend', orientation='v', y=1.1, x=1))
            
            fig.update_xaxes(
                tick0 = df.index[0],
                dtick=round(n*60*60*1000),
                tickformat="%Y-%m-%d %H",
                tickangle = 45)
            
            
            filename= title + '.html'
            
            fig.write_html(suspect_folder / filename)
            
            plot_count += 1
            
            
        #-- If there are flags for unrealistic pollutant comparisons specifically, plot this data together as well for more effective review (have to check for substrings since can have multiple Flagging_Reasons)
        # Note: these plots will not include other highlighted suspect or invalid data
        
        # Colors to use when plotting unrealistic pollutant comparisons to distinguish between two pollutants
        up_colrs = ['forestgreen','cornflowerblue']
        if any(df['Flagging_Reason'].str.contains('|'.join([unr_r for unr_r in unrealistic_pollutant_comparisons['flagging_reason']]))):
            
            for unr_r in unrealistic_pollutant_comparisons['flagging_reason']:
                
                unr_sites = df[df['Flagging_Reason']==unr_r][[config_dict['site_col'],config_dict['pollutant_col']]].drop_duplicates().reset_index(drop=True)

                for us in unr_sites[config_dict['site_col']].unique():
                    
                    # Define pollutants related to this unr flag
                    unr_ps = unr_sites.loc[unr_sites[config_dict['site_col']]==us,config_dict['pollutant_col']].tolist()
                    
                    fig = go.Figure()
            
                    
                    # Add red vertical lines to identify unrealistic data timestamps on plot
                    unr_dts = (df[(df[config_dict['site_col']]==us) & (df['Flagging_Reason']==unr_r)][config_dict['timestamp_col']]).unique()
                    # Annotations with datetime x for plotly vline have a bug, so instead plot first instance and add to legend and don't add rest
                    fig.add_vline(x=unr_dts[0], line_width=3, line_color="red", opacity=0.3,showlegend=True,name='Suspect')#layer='below')
                    # Add remaining vertical lines 
                    for d in range(1,len(unr_dts)):
                        unr_dt = unr_dts[d]
                        fig.add_vline(x=unr_dt, line_width=3, line_color='red', opacity=0.3)#layer='below')
                   
                    # Plot each pollutant
                    for ip in range(0,len(unr_ps)):
                        
                        up = unr_ps[ip]
                        up_colr = up_colrs[ip]                        
                        
                        p_df = df[(df[config_dict['site_col']]==us) & (df[config_dict['pollutant_col']]== up)]# & (df['Flagging_Reason']!=unr_r)]
                        
                        # Have to define customdata if want to show per value Suspect Reason (x, y or customdata recognized with %). <extra></extra> removes the hover legend text outside of the box that default appears
                        # TODO - add unit to hover text once added to config_dict
                        fig.add_trace(go.Scatter(x=list(p_df[config_dict['timestamp_col']]), y=list(p_df[config_dict['value_col']]),name=up, mode='lines+markers',marker_color=up_colr,hovertemplate=(f"{up}: " + "%{y}<extra></extra>")))
                          

                    # Add title and subtitles to                                            
                    title = f"Unrealistic Pollutant Behavior for {(' and '.join(unr_ps))} at {us}"
                    subtitle = '<br><sup>Note: Other Suspect or Invalid data is not highlighted in this figure.'
                    n = ((p_df[config_dict['timestamp_col']].max() - p_df[config_dict['timestamp_col']].min()).days*24)/10 # number of hours to space x tick marks by
                    
                    fig.update_layout(title=title+subtitle,
                                       xaxis_title=config_dict['timestamp_col'],
                                       yaxis_title='Hourly Data [unit placeholder]', #TODO add unit to y axis once thats added in config_dict
                                       legend=dict(title='Legend', orientation='v', y=1.1, x=1),
                                       hovermode="x unified") # one hover text per x axis point
                    
                    fig.update_xaxes(
                        tick0 = df.index[0],
                        dtick=round(n*60*60*1000),
                        tickformat="%Y-%m-%d %H",
                        tickangle = 45)
                                        
                    filename= title + '.html'
                    
                    fig.write_html(suspect_folder / filename)
                    
                    plot_count += 1
                
        print(f'{plot_count} plots of suspect data have been saved to subfolder Suspect Data for Review in {output_file_path}.',end='\n\n')
    
        # Ask user whether suspect data should be kept as is or removed (convert to lower case and remove any spaces user may have entered)
        #TODO - In future version, allow user to concur per Flagging Reason for suspect data (e.g., can choose to remove suspect sticking data but not suspect rate of change)
        print("Please review data flagged as 'S' (Suspect) in these plots before proceeding...")
        suspect_action = input("Should suspect data be removed? (Y - yes/N - no/X - exit)").lower().replace(' ','')

        # Keep asking user if answer is not an expected responses (user answer does not need to be case-sensitive)
        while not suspect_action in ['y','n','x']:
            
            print('Answer is not recognized. Please answer with the letter Y or N and press enter. Or enter X to exit program.')
            
            suspect_action = input("Should suspect data be removed? (Y/N/X)",end='\n\n')
            
        if suspect_action == 'y': 

            df.loc[df['QC_Flag']=='S',config_dict['value_col']] = np.nan
            
            print('All Suspect data removed from dataset. Suspect QC_Flags retained.',end='\n\n')
            
        elif suspect_action == 'n': 

            print('Suspect data kept as is in dataset. Suspect QC_Flags retained. Reminder: You can set your own thresholds for data flagging checks if default settings are unideal.',end='\n\n')
            
        elif suspect_action == 'x':
            
            print('Exiting program. Quality-controlled df will not be saved out. Reminder: You can set your own thresholds for data flagging checks if default settings are unideal.')
            sys.exit()
                    

    #-- Status statement, letting user know that data flagging is finished.
    print('Data have undergone data flagging review.',end='\n\n')
    

    #-- Summarize any major data flagging issues for output
       
    # Summarize data flagging (if I or S over 1% for pollutant/site pair) - show flag percent and associated list of unique flagging reasons 
    qc_tbl = (df[df['QC_Flag'].isin(['I','S'])]
                .groupby([config_dict['site_col'],config_dict['pollutant_col'],'QC_Flag'])
                .agg(**{'Flag_Count': (config_dict['value_col'], 'size'),
                        'Flagging_Reasons':('Flagging_Reason',lambda x: ','.join(x.unique()))})
                .reset_index()
                .merge(full_count,on=[config_dict['site_col'],config_dict['pollutant_col']])
                .assign(Flag_Percent = lambda x : round(x['Flag_Count']/x['Full_Count'] * 100,1))
                .drop(['Flag_Count','Full_Count'],axis=1)
                .loc[lambda x: x['Flag_Percent']>=1])
    
    # Move Flag_Percent to the 4th column
    re_org_col = qc_tbl.pop('Flag_Percent')
    qc_tbl.insert(3,'Flag_Percent',re_org_col)
    
    # Save out qc_tbl if not empty and print statement letting user know
    if not qc_tbl.empty:
        
        qc_tbl.to_csv(Path(output_file_path,'Data Flagging Highlights.csv'),index=False)
        
        print(f'Major data flagging issues (flags >= 1%) have been summarized in Data Flagging Highlights.csv and saved to {output_file_path}.')
        print('Recommend reviewing these site(s), pollutant(s) to diagnose cause of recurring data flags.',end='\n\n')

       
    #-- Organize and save out threshold table
    
    threshold_tbl = pd.DataFrame(threshold_tbl)
    threshold_tbl.insert(0,config_dict['pollutant_col'],threshold_tbl['Standard_Pollutant'].map(lambda pn: config_dict['pollutant_map'][pn]))
    threshold_tbl = threshold_tbl.sort_values(by=config_dict['pollutant_col']).drop('Standard_Pollutant',axis=1)
    threshold_tbl.to_csv(Path(output_file_path,'Data Flagging Thresholds.csv'),index=False)
    print(f"Thresholds for data flagging have been saved to {output_file_path}.")
    
    #-- Drop "Standard_Pollutant" name from returned df
    df = df.drop('Standard_Pollutant',axis=1)
        
    return df