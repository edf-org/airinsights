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



def data_flagging(input_data : pd.DataFrame,
                    config_dict : str,
                    range_thresholds : dict | None = None, 
                    roc_thresholds : dict | None = None,
                    sticking_hrs_thresholds : dict | None = None,
                    remove_invalid: bool = True,
                    remove_suspect: bool = False,
                    verbose_df : bool = False, 
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
    
    verbose_df : bool, default False
        If True, output DataFrame includes Flagging_Reason column, providing more information as to why data was flagged as Invalid or Suspect.
        
    output_file_path : str, optional
        The file path where output .txt files and figures should be saved. If None provided, current working directory will be used.
    
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing quality-controlled data with the site, timestamp, and pollutant columns from the input data and the following columns appended:
                
            **QC_Flag**: quality control flags - Invalid(I) and Suspect(S) 
                                
        If ``verbose_df`` is ``True``, also appends:
            
            **Flagging_Reason**: the check(s) or reason(s) that resulted in a QC_Flag
            
    thresholds_tbl (.csv saved out to output_file_path)
        A pandas DataFrame (long format) describing the checks performed per pollutant and their respective thresholds and data actions (Invalid or Suspect), with the following columns:
            
            **Pollutant**: pollutant in input_data
           
            **Flagging_Check**: flagging check
            
            **Invalid or Suspect**: data action associated from flagging check (I or S)
            
            **Threshold(s)**: thresholds used for specific flagging_check for associated pollutant 
            
        
    Notes
    -----
    
    If ``remove_suspect`` is ``False`` and Suspect data is found, 
        also saves out interative plotly time series (.html) of Suspect (and Invalid) data for review in new "Suspect Data for Review" folder in output_file_path.
    
    If Suspect or Invalid data makes up 1% or more of all data for a certain site and pollutant, 
        a "Data Flagging Evaluation Summary for Review_[runtime date].txt" file is saved to output_file_path with this summarized information and guidance for review.


    """


    # Defining internal variables for further flagging (no user-defined allowed) and plotting features
    # Unrealistic pollutant comparisons, unr_ps: list of [x,y]  where unrealistic pollutant check is for x > y, and Flagging_Reason code needs to match order of [x,y]}
    # Note: this does not include unrealistic pollution behvaior for O3 which is defined separately in the code below
    unr_pollutant_comparisons = {'unr_ps':[['PM2_5','PM10'],['NO2','NOx'],['NO','NOx']],'flagging_reason':['UP','UN','UN']}
    # Colors to use when plotting unrealistic pollutant comparisons to distinguish between two pollutants
    up_colrs = ['forestgreen','cornflowerblue']
    
    default_range_thresholds = {'BC':{'min':-400,'max':500000},'CO':{'min':0,'max':15},'NO2':{'min':-1,'max':500},
                                      'NO':{'min':-1,'max':3000},'O3':{'min':-5,'max':250},'PM2_5':{'min':-5,'max':1000},
                                      'PM10':{'min':-5,'max':10000},'SO2':{'min':-5,'max':300},'VOC':{'min':-1,'max':300}}
    
    # Generic range check option: common error codes (unitless)
    common_error_codes = [-9999, 865, 895, 995, 985, 999, 999.9, 985, 9999, 99999,1000,2000,3000,4000,5000,6000,7000,8000,9000]
    
    default_roc_thresholds = {'NO2':50,'O3':60,'PM2_5':150,'PM10':300}
    
    default_sticking_hrs_thresholds = {'NO2':{'stck_hr_count':10,'min_val':5},'O3':{'stck_hr_count':12,'min_val':10},
                                             'PM2_5':{'stck_hr_count':6,'min_val':5},'PM10':{'stck_hr_count':6,'min_val':10}}
    
    
    #--Check format of thresholds and other input fields and set default thresholds where not defined
    
    # Range thresholds
    if range_thresholds is None:
        range_thresholds = default_range_thresholds
        
    elif not isinstance(range_thresholds, dict): 
        raise TypeError(f"dictionary expected for range_thresholds, got {type(range_thresholds).__name__}")
    
    # ROC thresholds   
    if roc_thresholds is None:
        roc_thresholds = default_roc_thresholds
        
    elif not isinstance(roc_thresholds, dict): 
        raise TypeError(f"dictionary expected for roc_thresholds, got {type(roc_thresholds).__name__}")

    
    # Stickhing thresholds
    if sticking_hrs_thresholds is None:
        sticking_hrs_thresholds = default_sticking_hrs_thresholds
        
    elif not isinstance(sticking_hrs_thresholds, dict): 
        raise TypeError(f"dictionary expected for sticking_hrs_thresholds, got {type(sticking_hrs_thresholds).__name__}")

    # Output_file_path
    if not isinstance(output_file_path, str): 
        raise TypeError(f"string expected for output_file_path, got {type(output_file_path).__name__}")
    
          
    # Make a copy of the user input data and check if hourly (will raise an error if data only contains non-hourly freq or will print out how much data needed to be excluded)
    df = input_data.copy()
    df = _validate_hourly(df,config_dict)
    
    # Subset config_dict to pollutants that are actually in the dataset
    p_in_df = [pp for pp in config_dict['pollutant_map'].keys() if config_dict['pollutant_map'][pp] in df[config_dict['pollutant_col']].unique()]
    config_dict['pollutant_map'] = {k: config_dict['pollutant_map'][k] for k in p_in_df}
            
    # Initiate table where thresholds will be saved out
    threshold_tbl = pd.DataFrame(columns = ['Pollutant','Flagging_Check','Invalid or Suspect','Threshold(s)'])
         
    # Status statement, letting user know what stage the data audit function is at 
    print('Data are undergoing data flagging review....',end='\n\n')
    
    # Add flagging-related columns 
    df = df.sort_values(by=[config_dict['site_col'],config_dict['pollutant_col'],config_dict['timestamp_col']]).reset_index(drop=True)
    df['QC_Flag'] = '' #Create new column in df for QC Flags (initially blank)
    df['Flagging_Reason'] = '' # Create new column in df for Suspect Reasons (initially blank)

    # Count number of non-NaN entries per site,pollutant pair - to be used later to calculate % of data with QC flags 
    # Have to do this before, since rest of program may nan invalid/suspect data which would bias the count low)
    # Use count() instead of size() to only consider non NA values in denominator later on
    group_cols = [c for c in df.columns if c not in [config_dict['value_col'], config_dict['timestamp_col'],'QC_Flag','Flagging_Reason']]
    full_count = df.groupby([config_dict['site_col'],config_dict['pollutant_col']])[config_dict['value_col']].count().reset_index().rename(columns = {config_dict['value_col'] :'Full_Count'})
    
    
    # Loop through pollutant_map in config_dict to check their formats and run invalid checks
    for p in config_dict['pollutant_map'].keys():
        
        # Define pollutant name as exists in df to subset appropriately in checks
        p_name = config_dict['pollutant_map'][p]
        
        # Grab pollutant-specific thresholds
        
        # If p is missing, check if threshold exists for p in default, otherwise use generic (set to None initially and generic is used later)
        if p not in range_thresholds.keys():
            
            try:
                
                p_range = default_range_thresholds[p] 
                
            except:
            
                p_range = None # this triggers use of common error codes later
        
        # Otherwise, assign p_range from set thresholds and check format
        else:
            
            p_range = range_thresholds[p] 

            if not isinstance(p_range, dict): 
                raise TypeError(f"Dictionary with range min and max expected, got {type(p_range).__name__} for {p_name}.")
                
            elif 'min' not in p_range.keys() or 'max' not in p_range.keys():
                raise KeyError(f"Missing keys for {p_name} range threshold. Both min and max keys needed per pollutant.")
            
            # Check that min,max in p_range provided correctely (max>min)
            elif p_range['min'] >= p_range['max']:
                raise ValueError(f"Issue with user-provided range threshold for {p_name}. Min is greater than or equal to max.")
        
        # Perform Range Check - Invalid
        if p_range:
        
            # Apply "I" QC_Flag and "RG" Flagging_Reason
            df.loc[(df[config_dict['pollutant_col']]==p_name) & ((df[config_dict['value_col']]<p_range['min']) | (df[config_dict['value_col']]>p_range['max'])),['QC_Flag', 'Flagging_Reason']] = ['I','RG']
        
        # Perform Range Check - Invalid with common error codes
        else:
            
            # Apply "I" QC_Flag and "RG" Flagging_Reason
            df.loc[(df[config_dict['pollutant_col']]==p_name) & (df[config_dict['value_col']].isin(common_error_codes)),['QC_Flag', 'Flagging_Reason']] = ['I','RG']
            
            print(f'Warning: No defined min/max range for pollutant {p} in default or user-defined settings. Data reviewed against common error codes instead. See output thresholds summary.\nReminder: you can specify user-defined thresholds.')

        
        # Add p_range threshold to thresholds_tbl
        #TODO - add units to thresholds where appropriate and convert
        threshold_tbl = pd.concat([threshold_tbl,pd.DataFrame([{'Pollutant':p_name,'Flagging_Check':'Range','Invalid or Suspect':'I','Threshold(s)':p_range}])],axis=0)

    
    # Leave pollutant for loop to invalidate data and calculate ROC and ROC_rolling accurately (these calculations will be incorrect if performed earlier and invalid data is removed later)
    
    # Keep dataframe with all data for plotting later!
    full_df = df.copy()
    
    # By default, nan invalid data. If False, keep.
    if remove_invalid:
        
        print('Invalid data being removed based on user input.',end='\n\n')
        
        df.loc[df['QC_Flag']=='I',[config_dict['value_col'],'ROC']] = np.nan
    
    # Calculate rate of change (ROC) and rolling ROC after invalid data checks  - per parameter and site
    df['ROC'] = abs(df.groupby(group_cols)[config_dict['value_col']].diff().reset_index(drop=True))    
    df['ROC_roll'] = df.groupby(group_cols)['ROC'].transform(lambda r: r.rolling(window=2, min_periods=1).mean().shift(-1))
    
    
    
    #--Loop through pollutant_map in config_dict to check their formats and run suspect checks
    
    for p in config_dict['pollutant_map'].keys():
        
        # Define pollutant name as exists in df to subset appropriately in checks
        p_name = config_dict['pollutant_map'][p]
        
        # Set pollutant-specific Suspect thresholds
        
        # Rate of Change thresholds
        
        # If p missing from thresholds, try default and then use generic rate of change threshold (10x mean)
        if p not in roc_thresholds.keys():

            try:
                p_roc = default_roc_thresholds[p]
            
            except:
            
                p_roc = 10 * df.loc[df[config_dict['pollutant_col']]==p_name,config_dict['value_col']].mean()
                print(f'Warning: No defined rate of change for pollutant {p} in default or user-defined settings. Data reviewed against 10x(pollutant mean) instead. See output thresholds summary.\nReminder: you can specify user-defined thresholds.',end='\n\n')
            
        # Otherwise, assign p_roc from set thresholds and check format
        else:     
            
            p_roc = roc_thresholds[p]
        
            if not isinstance(p_roc, int) and (not isinstance(p_roc,float)): 
                raise TypeError(f"Integer or float expected for rate of change, got {type(p_roc).__name__} for {p_name}.")
                
            elif p_roc<0:
                raise ValueError(f"Absolute rate of change value cannot be negative. Fix threshold for {p_name}.")
            
        
        # Sticking thresholds
        
        # If p missing from thresholds, try default and then use generic
        if p not in sticking_hrs_thresholds.keys():
            
            try:
                p_sticking = default_sticking_hrs_thresholds[p]
            
            except:
                p_sticking = {'stck_hr_count':12,'min_val':0}
                print(f'Warning: No defined sticking threshold for pollutant {p} ({p_name}) in default or user-defined settings. Data reviewed against generic sticking threshold instead. See output thresholds summary.\nReminder: you can specify user-defined thresholds.',end='\n\n')

        
        # Otherwise, assign p_sticking from set thresholds and check format
        else:
            
            p_sticking = sticking_hrs_thresholds[p]
        
            if not isinstance(p_sticking, dict): 
                raise TypeError(f"Dictionary with stck_hr_count and min_val expected in sticking_hr_thresholds, got {type(p_sticking).__name__} for {p_name}.")
                
            elif 'stck_hr_count' not in p_sticking.keys() or ('min_val' not in p_sticking.keys()):
                raise KeyError(f"Missing keys for {p_name} sticking_hr_thresholds. Both stck_hr_count and min_val keys needed per pollutant.")
            
        
        #TODO - add units to thresholds where appropriate and convert
        # Add these thresholds into threshold_tbl
        threshold_tbl = pd.concat([threshold_tbl,pd.DataFrame([{'Pollutant':p_name,'Flagging_Check':'Rate of Change','Invalid or Suspect':'S','Threshold(s)':p_roc}])],axis=0)
        threshold_tbl = pd.concat([threshold_tbl,pd.DataFrame([{'Pollutant':p_name,'Flagging_Check':'Sticking','Invalid or Suspect':'S','Threshold(s)':p_sticking}])],axis=0)

        
        #  Perform Rate of Change Check - Suspect
        # Ok to use loc here since first time assigning S Suspect data for this pollutant and making sure to not overwrite any 'I' flag data (which could happen if user set remove_invalid = False)
        df.loc[((df[config_dict['pollutant_col']]==p_name) & (df['ROC_roll']>p_roc) & (df['QC_Flag'] != 'I')), ['QC_Flag', 'Flagging_Reason']] = ['S', 'RC']
        
    
        # Perform Sticking Check - Suspect
        df['Sticking_Count'] = df.groupby((df['ROC']!=0).cumsum()).transform('size')
        
        # Use np.where to not overwrite any existing Flagging_Reasons. Making sure to not overwrite 'I' flag, which could happen if user set remove_invalid = False
        p_sticking_idx = (df[config_dict['pollutant_col']]==p_name) & (df['Sticking_Count']>=p_sticking['stck_hr_count']) & (df[config_dict['value_col']]>=p_sticking['min_val'])
        df.loc[p_sticking_idx & (df['QC_Flag'] != 'I'),'QC_Flag'] = 'S'
        df['Flagging_Reason'] = np.where(p_sticking_idx & (df['QC_Flag'] != 'I'), df['Flagging_Reason'] + 'ST', df['Flagging_Reason'])
        df.drop('Sticking_Count',axis=1,inplace=True) 
            
            
    # Continue with checks that don't need to occur within pollutant 'for loop'...
     
    # Perform "Unrealistic Pollution Behavior" checks for relevant pollutants
    
    # Perform Unrealistic Pollution Behavior check for O3 separately, since different format from others (and uses ROC column)
    
    if 'O3' in config_dict['pollutant_map'].keys():
        
        unr_o3_condition = ((df[config_dict['pollutant_col']]==config_dict['pollutant_map']['O3']) & ((df[config_dict['timestamp_col']].dt.hour >= 23) | (df[config_dict['timestamp_col']].dt.hour <= 6)) & ((df[config_dict['value_col']]>70) | (df['ROC']>40)))
        # Making sure to not overwrite 'I' flag, which could happen if user set remove_invalid = False
        df.loc[unr_o3_condition & (df['QC_Flag'] != 'I'),'QC_Flag'] = 'S'
        df['Flagging_Reason'] = np.where(unr_o3_condition & (df['QC_Flag'] != 'I'), df['Flagging_Reason'] + 'UO', df['Flagging_Reason'])
        
        threshold_tbl = pd.concat([threshold_tbl,pd.DataFrame([{'Pollutant':config_dict['pollutant_map']['O3'],'Flagging_Check':'Unrealistic Pollutant Behavior','Invalid or Suspect':'S','Threshold(s)':'See Documentation..'}])],axis=0)

        
    # Drop ROC column
    df.drop(['ROC','ROC_roll'],axis=1,inplace=True)
    
    
    # Perform remaining "Unrealistic Pollution Behavior" checks - defined in unr_pollutant_comparisons (all work better with wide data format)
    
    for i in range(0,len(unr_pollutant_comparisons['unr_ps'])):
        
        unr_lst = unr_pollutant_comparisons['unr_ps'][i]
        unr_sr = unr_pollutant_comparisons['flagging_reason'][i]
        
        if all(unr_p in config_dict['pollutant_map'].keys() for unr_p in unr_lst):
            
            # Pivoting wider to find site + timestamp pairs where certain pollutants are showing unrealistic pollutant behavior - just using df_pivot to find indexes, not melting this dataframe back to df (since QC_Flag and Flagging_Reason columns are pollutant-specific and don't work in wide format)
            index_cols = [c for c in df.columns if c not in [config_dict['pollutant_col'],config_dict['value_col']]]
            index_cols.remove('QC_Flag') #remove since will be inaccurate in df_pivot
            index_cols.remove('Flagging_Reason') #remove since will be inaccurate in df_pivot
            df_pivot = df.pivot(index=index_cols,columns=config_dict['pollutant_col'],values = config_dict['value_col']).reset_index()

                
            # Get site + timestamp pair where unrealistic pollutant comparison (x>y)
            unr_sites_dates = pd.MultiIndex.from_frame(df_pivot[(df_pivot[config_dict['pollutant_map'][unr_lst[0]]] > df_pivot[config_dict['pollutant_map'][unr_lst[1]]])][[config_dict['site_col'],config_dict['timestamp_col']]])
            
            # Find rows in df where site and datetime pair equal those identified in unr_sites_dates AND pollutant column is equal to either pollutants in unr_lst
            unr_lst_names = [config_dict['pollutant_map'][k] for k in unr_lst]
            unr_idx = pd.MultiIndex.from_frame(df[[config_dict['site_col'],config_dict['timestamp_col']]]).isin(unr_sites_dates) & df[config_dict['pollutant_col']].isin(unr_lst_names)
            
            # Flag unrealistic pollutant rows as Suspect and add Suspect Reason to exisiting column. Making sure to not overwrite 'I' flag, which could happen if user set remove_invalid = False
            df.loc[unr_idx & (df['QC_Flag'] != 'I'),'QC_Flag'] = 'S'  
            df['Flagging_Reason'] = np.where(unr_idx & (df['QC_Flag'] != 'I'), df['Flagging_Reason'] + unr_sr, df['Flagging_Reason'])
            
            threshold_tbl = pd.concat([threshold_tbl,pd.DataFrame([{'Pollutant':config_dict['pollutant_map'][unr_lst[0]],'Flagging_Check':'Unrealistic Pollutant Behavior','Invalid or Suspect':'S','Threshold(s)':f"{config_dict['pollutant_map'][unr_lst[0]]}>{config_dict['pollutant_map'][unr_lst[1]]}"}])],axis=0)

            threshold_tbl = pd.concat([threshold_tbl,pd.DataFrame([{'Pollutant':config_dict['pollutant_map'][unr_lst[1]],'Flagging_Check':'Unrealistic Pollutant Behavior','Invalid or Suspect':'S','Threshold(s)':f"{config_dict['pollutant_map'][unr_lst[0]]}>{config_dict['pollutant_map'][unr_lst[1]]}"}])],axis=0)

    
    # Remove suspect data = True, then nan any data where QC_Flag is 'S'
    if remove_suspect:
        
        print('Suspect data being removed based on user input.',end='\n\n')
                
        df.loc[df['QC_Flag']=='S',config_dict['value_col']] = np.nan
        
        suspect_action = 'y'
    
    
    # Else if any Suspect data in dataframes and remove_suspect=False, create suspect data plots for user to review and ask for concurrence on removing/keeping
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
            
            
        # If there are flags for unrealistic pollutant comparisons specifically, plot this data together as well for more effective review (have to check for substrings since can have multiple Flagging_Reasons)
        # Note: these plots will not include other highlighted suspect or invalid data
        if any(df['Flagging_Reason'].str.contains('|'.join([unr_r for unr_r in unr_pollutant_comparisons['flagging_reason']]))):
            
            for unr_r in unr_pollutant_comparisons['flagging_reason']:
                
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
                    

            
    # Summarize data flagging  (if over 1% for pollutant/site combo)
    
    qc_count = df[df['QC_Flag'].isin(['I','S'])].groupby([config_dict['site_col'],config_dict['pollutant_col']])['QC_Flag'].value_counts().reset_index().rename(columns = {'count':'Flag_Count'})
    qc_count = pd.merge(full_count,qc_count,on=[config_dict['site_col'],config_dict['pollutant_col']])
    qc_count['Flag_Perct'] = round((qc_count['Flag_Count']/qc_count['Full_Count']) * 100,1)
    
    # Status statement, letting user know what stage the data audit function is at
    print('Data have undergone data flagging review.',end='\n\n')
    
    # Invalid and Suspect % data summaries will print regardless or remove_invalid or remove_suspect/suspect_action
    #TODO - Add % breakdown of flagging reason into these summaries. This will be especially helpful for Suspect data.
    
    if not qc_count[qc_count['Flag_Perct']>=1].empty:
        
        # Create text file to save out data flagging % summaries
        summary_text_file = Path(output_file_path,'Data Flagging Evaluation Summary for Review.txt')
        open(summary_text_file,'w')
        
        print(f"Data Flagging Evaluation Summary has been saved to {output_file_path}.",end='\n\n')
    
        if not qc_count[(qc_count['Flag_Perct']>=1) & (qc_count['QC_Flag']=='I')].empty:
            
            with open(summary_text_file,'a') as f:
            
                print('Printing site,pollutant combinations that were flagged Invalid for 1% or more of data...',file=f)
        
                for _,row in qc_count[(qc_count['Flag_Perct']>=1) & (qc_count['QC_Flag']=='I')].iterrows():
                    
                    print(f"{row['Flag_Perct']}% of {row[config_dict['pollutant_col']]} at {row[config_dict['site_col']]} flagged as Invalid",file=f)
                    
                print('Recommend reviewing these site(s), pollutant(s) to diagnose cause of recurring data issues.',end='\n\n',file=f)
     
            
        if not qc_count[(qc_count['Flag_Perct']>=1) & (qc_count['QC_Flag']=='S')].empty: # & (suspect_action=='y'):
            
            with open(summary_text_file,'a') as f:
    
                print('Printing site,pollutant combinations that were flagged Suspect for more than 1% of data...',file=f)
        
                for _,row in qc_count[(qc_count['Flag_Perct']>=1) & (qc_count['QC_Flag']=='S')].iterrows():
                    
                    print(f"{row['Flag_Perct']}% of {row[config_dict['pollutant_col']]} at {row[config_dict['site_col']]} flagged as Suspect",file=f)
                    
                print('Recommend reviewing these site(s), pollutant(s) to diagnose cause of recurring suspect data.',end='\n\n',file=f)
            
    else:
        print('Invalid or suspect data issues are not persistent. No instances of site,pollutant combinations where data was flagged as Invalid for 1% or more of data.',end='\n\n')

     
    
                                                                             
    if not verbose_df:
        
        df.drop('Flagging_Reason',axis=1,inplace=True) 
       
    # Save out table of qc thresholds
    threshold_tbl = threshold_tbl.sort_values(by='Pollutant') 
    threshold_tbl.to_csv(Path(output_file_path,'Data_Flagging_Thresholds.csv'),index=False)
    print(f"Thresholds for data flagging have been saved to {output_file_path}.")
        
    return df