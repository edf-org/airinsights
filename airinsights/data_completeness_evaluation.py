#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:04:41 2026


"""
"""
@author: TD Enviro
"""

import pandas as pd
import numpy as np
import seaborn as sns
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib import transforms
import matplotlib.patches as patches
import warnings
from airinsights.helpers import _validate_hourly
from internal_data_completeness_functions import _daily_comp, _month_comp, _annual_comp



def data_completeness_evaluation(input_data : pd.DataFrame,
                    config_dict : str,
                    comp_threshold : int | float | None = 75, 
                    month_gap_test : bool = True,
                    daily_comp_vis : bool = False,
                    output_file_path : str | None = os.getcwd()
                    ):
    """Comprehensive data completeness evaluation based on comp_threshold, showing site and network-wide completeness and coinciding data use guidance.


    
    Parameters
    ----------
    input_data : pd.DataFrame
        A pandas DataFrame containing hourly AQ data (read in using read_aqdata_file)
    
    config_dict : dict
        A dictionary detailing the input_data structure, containing input parameter names and values. 
    
    comp_threshold : int, optional
        Data completeness threshold (%) to use for daily, monthly, and annual completeness calculations. Default is 75%, comp_threshold = 75.
        
     month_gap_test : bool, default True
         Includes requirement that no more than 2 consecutive months can be missing per year to the annual completeness requirement (as a proxy for seasonal completeness).
         If False, does not require this condition to be met.
    
    daily_comp_vis : bool, default False
        If True, output also includes time series of # of sites with complete daily data per pollutant across the DataFrame time period.
        Will be set to True if any daily data completeness issues are found (e.g., variable daily data completeness)
        
    output_file_path : str, optional
        The file path where output .txt files and figures should be saved. If None provided, current working directory will be used.
        
    
    Returns
    -------
    
    annual_df : pd.DataFrame
        A pandas DataFrame of annual completeness per year, site, and pollutant.
    
    Site-specific monthly and annual data completeness heatmap per pollutant (.jpg's saved to output_file_path)
        Heatmap visualizes site data completeness with rows = site and columns = time period (months, years)
    
    Network-wide data completeness heatmap per pollutant - ONLY if more than one site in DataFrame (.jpg's saved to output_file_path)
        Heatmap visualizes network-wide data completeness with rows = pollutant, columns = time period (months, years), and cell = % of network with complete data
    
    Data Completness Evaluation Summary (.txt file saved to output_file_path)
        Text file containing data use guidance based on annual network completeness and summary of any daily network-wide completeness issues found.
        
    Notes
    -----
    
    If ``daily_comp_vis`` is ``True`` or any network-wide daily data completenss issues are found, also saves out daily network-wide completeness time series to output_file_path


    """
    
    #-- Define internal functions
    
    
    # Base figsize and fontsize on the number of rows and columns in dataframe - can get unruly with lots of sites or years for the heatmaps
    def _figsize(df, cell_width=0.8, cell_height=0.6, min_size=(6,4), max_size=(30, 20)):
        n_rows, n_cols = df.shape
        width = max(min_size[0], min(max_size[0], n_cols * cell_width))
        height = max(min_size[1], min(max_size[1], n_rows * cell_height))
        return (width, height)
    
    def _fontsize(df,base_fontsize=12,min_fontsize=6,max_fontsize=12):
        n_rows, n_cols = df.shape
        scale_factor = max(n_rows, n_cols)
        fontsize = base_fontsize * (10 / scale_factor)  # 10 = "reference" grid size
        fontsize = max(min_fontsize, min(max_fontsize, fontsize))
        return fontsize
    
    # Set network completeness category based on range of %'s
    def _network_category(d):
        
        if (d>=0) & (d<.25):
            cat = 0
        elif (d>=.25) & (d<.50):
            cat = 1
        elif (d>=.50) & (d<.75):
            cat=2
        elif (d>=.75):
            cat=3
        else:
            cat = np.nan
        
        return(cat)
    
    # Site-specific completeness color (0-incomplete,1-complete)
    comp_colors = {0:'white',1:'forestgreen'}
    
    # Completeness category colors
    network_cat_colors = {0:'white',1:'lightsteelblue',2:'royalblue',3:'navy'}
    network_cat_texts = {0:'<25%',1:'25-49%',2:'50-74%',3:'>75%'}
    
    # Create network completeness legend features
    network_legend_shps = []
    network_legend_txts = []
    
    for cr in network_cat_colors.keys():
    
        network_legend_shps.append(patches.Rectangle((0,0),1,1,facecolor=network_cat_colors[cr],edgecolor='black',linewidth=0.5))
        network_legend_txts.append(network_cat_texts[cr])
     
    # Reverse legend features so going from low to high
    network_legend_shps.reverse()
    network_legend_txts.reverse()

    
    #-- Check format of input fields  
    
    # Check that completeness threshold is within accepted 60-100% range and that it's the correct format
    if comp_threshold < 60 or comp_threshold > 100:
        raise ValueError("comp_threshold can only be set between 60-100%.")
       
    elif not isinstance(comp_threshold, int) | isinstance(comp_threshold, float): 
        raise TypeError(f"integer or float expected for comp_threshold, got {type(comp_threshold).__name__}")
    
    # Output_file_path
    if not isinstance(output_file_path, str): 
        raise TypeError(f"string expected, got {type(output_file_path).__name__}")
     
        
    # Status statement, letting user know what stage the data audit function is at
    print('Data are undergoing data completeness review....')
        
    # Make a copy of the user input data and check if hourly (will raise an error if data only contains non-hourly freq or will print out how much data needed to be excluded)
    hr_df = input_data.copy()
    hr_df = _validate_hourly(hr_df,config_dict)
    
    # Subset config_dict to pollutants that are actually in the dataset
    p_in_df = [pp for pp in config_dict['pollutant_map'].keys() if config_dict['pollutant_map'][pp] in hr_df[config_dict['pollutant_col']].unique()]
    config_dict['pollutant_map'] = {k: config_dict['pollutant_map'][k] for k in p_in_df}
    
    # Check how many days are in the dataset
    total_day_count = (hr_df[config_dict['timestamp_col']].max() - hr_df[config_dict['timestamp_col']].min()).days
    
    # Warn user that if dataset contains less than half a year of data, the annual completeness evaluations may not be informative. If dataset contains less than a month of data, stop code with ValueError
    if (total_day_count < 365/2) and (total_day_count >= 28):
        
        warnings.warn("Dataset contains less than half a year of data. Annual completeness evaulations may not be informative.")
        
    elif total_day_count < 28:
        
        raise ValueError('Dataset contains insufficient data for data completeness evaluation (less than 1 month).')
    
    # IF QC_Flag and Flagging_Reason column are present in the data, remove them, since this will mess up the groupby functions in _daily_comp,_month_comp,_annual_comp
    hr_df = hr_df.drop(['QC_Flag','Flagging_Reason'],axis=1,errors='ignore')
    
    # Use internal comp functions to create daily, monthly, and annual 
    print('Data are undergoing daily, monthly, and annual averaging...',end='\n\n')
    day_df = _daily_comp(hr_df,config_dict,comp_threshold) 
    month_df = _month_comp(day_df,config_dict,comp_threshold)
    annual_df = _annual_comp(month_df,config_dict,comp_threshold,month_gap_test)
    
    # Set nested dictionary of dataframes and timestamp formats to use in plotting for loop
    viz_periods = {'Monthly':{'df':month_df,'timestamp_format':'%m/%Y'},'Annual':{'df':annual_df,'timestamp_format':'%Y'},'Daily':{'df':day_df,'timestamp_format':'%Y-%m-%d'}}
    
    # Get features of the dataframe (overall, not pollutant-specific)
    total_site_count = hr_df[config_dict['site_col']].nunique()
    total_year_count = hr_df[config_dict['timestamp_col']].dt.year.nunique()


    # Let user know that output figures may be hard to read if very large # of sites or large # of years
    if total_site_count > 100:
        
        print('Output figure(s) may be hard to read due to large # of sites in input dataset.',end='\n\n')
        
    if total_year_count > 3:
        
        print('Output monthly completeness figure(s) may be hard to read due to multi-year span of input dataset.',end='\n\n')
        
    
    # If more than one site in input_data, create network max size per pollutant to reference later for network %'s
    if total_site_count > 1: 
        
        network_size = day_df.groupby(config_dict['pollutant_col'])[config_dict['site_col']].nunique().reset_index().rename(columns = {config_dict['site_col']:'Total_Site_Count'})
        
    # Create text file to save out data completeness print statements
    summary_text_file = Path(output_file_path, 'Data Completeness Evaluation Summary.txt')
    open(summary_text_file,'w')
    
    #TODO - should monthly completeness heatmaps include all possible calendar months, even if not in dataset? This is included for annual completeness checks, but not visualized. Currently only data from dataset is shown in monthly completeness
    # Loop through each averaging period and create completeness heatmaps and provide data use guidance
    for period in viz_periods.keys():
        
        # Read in correct dataframe from nested viz dictionary and get timestamp format
        df = viz_periods[period]['df']
        timestamp_format = viz_periods[period]['timestamp_format']
        
        # Create Comp column that indicates 1 - Complete, 0 - Incomplete data based on isna
        df['Comp'] = (~df[config_dict['value_col']].isna()).astype(int)
        
        # If the period is not daily, make site-specific heat maps of completeness
        if period != 'Daily':
        
            # Create site-specific data completeness heatmaps per pollutant (rows = sites, columns = time period)
            for p in config_dict['pollutant_map'].keys():
                
                # Define pollutant name as exists in df to subset appropriately in checks
                p_name = config_dict['pollutant_map'][p]
                
                # Subset df to pollutant based on user-defined name
                dfp = df[df[config_dict['pollutant_col']] == p_name].reset_index(drop=True)
                dfp = dfp.pivot(index=config_dict['site_col'],columns=config_dict['timestamp_col'],values='Comp')
                # Pivot and unpivot to make sure all sites are present in each heatmap (even if they don't have data, for consistency, fill NaN with zero)
                dfp = dfp.reset_index().melt(id_vars=config_dict['site_col'],value_vars=dfp.columns,var_name=config_dict['timestamp_col'],value_name='Comp')
                dfp['Comp'] = dfp['Comp'].fillna(0)
                dfp[config_dict['timestamp_col']] = pd.to_datetime(dfp[config_dict['timestamp_col']])
                
                # Get unique comp values to assign heatmap colors correctly
                comps = dfp['Comp'].unique().astype(int).tolist()
                comps.sort()
                
                #-- Make figures per reasonable group size (12 months, 12 years)
                
                dfp['year'] = dfp[config_dict['timestamp_col']].dt.year
                
                # Group by years
                if period=='Monthly':
                
                    dfp = dfp.rename(columns = {'year':'grouping'})
                
                # Group by 12 year chunks
                elif period=='Annual':
                    
                    dfp['grouping'] = (dfp["year"] - dfp["year"].min()) // 12 + 1
                
                for g in dfp['grouping'].unique():

                    # Pivot dataframe for easier heatmap plotting
                    dfp_f = (dfp[dfp['grouping']==g]
                             .pivot(index=config_dict['site_col'],columns=config_dict['timestamp_col'],values='Comp'))
                    
                    # Change formatting of columns to more readable datatime formats based on period
                    dfp_f.columns = [c.strftime(timestamp_format) for c in dfp_f.columns]
                    
                    fontsize = _fontsize(dfp_f)

                    title_start = dfp_f.columns[0]
                    title_end = dfp_f.columns[-1]
                    
                    if title_start != title_end:
                        title_period = f'{title_start} to {title_end}'
                    else:
                        title_period = title_start
                    
                    # Create heatmap
                    fig,ax = plt.subplots(figsize=(_figsize(dfp_f)))
                    plt.rcParams.update({"font.family" : "serif",'font.serif':['Arial']})
                    
                    sns.heatmap(dfp_f,cmap=[comp_colors[c] for c in comps],cbar=False,linewidths=0.25,linecolor='black',yticklabels=1)
                    plt.ylabel('')
                    plt.xlabel('')
                    plt.xticks(fontsize=14,rotation=45)
        
                    # Create a custom legend defining complete and incomplete data colors
                    rect1 = patches.Rectangle((0,0),1,1,facecolor='forestgreen',edgecolor='black',linewidth=0.5)
                    rect2 = patches.Rectangle((0,0),1,1,facecolor='white',edgecolor='black',linewidth=0.5)
                    rect3 = patches.Rectangle((0,0),1,1,facecolor='white',edgecolor='white',linewidth=0.5) #placeholder for completeness text
                    plt.legend((rect1,rect2,rect3), ('Complete', 'Incomplete',f'Completeness\nthreshold: {comp_threshold}%'),loc='center left',bbox_to_anchor=(1, 0.5),fontsize=14)
                   
                    # Title and save out
                    title = f'{period} {p_name} Site Completeness'
                    plt.title(title + f'\n{title_period}',fontsize=16)
                    plt.tight_layout()
                    plt.savefig(Path(output_file_path,title + f" {title_period.replace('/','_')}.png"),dpi=300)
                    plt.close(fig)
                
                # Pivot full dataframe for rest of code in this for loop
                dfp_v = dfp.pivot(index=config_dict['site_col'],columns=config_dict['timestamp_col'],values='Comp')
                # Change formatting of columns to more readable datatime formats based on period
                dfp_v.columns = [c.strftime(timestamp_format) for c in dfp_v.columns]

                # Provide data use guidance based on data completeness per year/across multiple years if period is Annual
                if period == 'Annual':
                    
                    with open(summary_text_file,"a") as f:
                    
                        print(f'{p_name} analysis guidance per year based on annual data completeness...',end='\n\n',file=f)
                        
                        # Get number and list of sites that are complete per year and provide data use guidance on seasonal analysis
                        for c in dfp_v.columns:
                            
                            comp_sites = dfp_v[dfp_v[c]==1].index.tolist()
                            
                            total_p_sites = network_size['Total_Site_Count'].loc[network_size[config_dict['pollutant_col']]==p_name].values[0]
                            
                            if len(comp_sites)>=1:
                            
                                print(f'- {c}: {len(comp_sites)} site(s) out of {total_p_sites} site(s) in your network have enough data this year for {p_name} seasonal analysis.',file=f)
                                
                                # Determine whether there are enough complete sites in the year for spatial hotspot analysis
                                if len(comp_sites)>=5:
                                    
                                    print('There are also enough complete sites in the network for spatial hotspot analysis (>=5).',file=f)
                                    
                                else:
                                    print('Spatial hotspot analysis is not recommended (>=5 complete sites needed).',file=f)
                                    
                                print(f'Sites: {comp_sites}',file=f,end='\n\n')
                                
                            else:
                                
                                print(f'- {c} - no sites meet the data completeness threshold ({comp_threshold}%). Seasonal analysis and spatial hotspot analysis for {p_name} are not advised due to major network data completeness issues. Analysis should be limited to specific periods with sufficient data. See output Monthly completeness plots.',file=f,end='\n\n')
                        # If there are multiple pollutants
                        
                        
                        # If there are multiple years, also provide data use guidance on long-term trend analysis
                        if len(dfp_v.columns)>1:
                            
                            # Get list of years
                            years = dfp_v.columns.tolist()
                        
                            # Look for sites that are complete for each year of the dataset
                            dfp_v['Multi-Year Comp'] = dfp_v.sum(axis=1)/len(dfp_v.columns)
                            m_comp_sites = dfp_v[dfp_v['Multi-Year Comp']==1].index.tolist()
                            
                            if len(comp_sites)>1:
                                
                                print(f'{len(m_comp_sites)} sites(s) out of {total_p_sites} site(s) in your network have enough data across all years {years} for long-term trend analysis. Sites: {m_comp_sites})',file=f,end='\n\n')
                                
                            else:
                                
                                print(f'No site(s) meet the data completeness threshold ({comp_threshold}%) across all years {years}. Long-term trend analysis is not advised due to major network data completeness issues. Analysis should be limited to years/specific periods with sufficient data. See output Annual and Monthly completeness plots.',file=f,end='\n\n')
                    
                        print('IMPORTANT: Review data use guidance from performance_evaluation() tool (bias and precision information) before proceeding with analysis.',end='\n\n',file=f)
                
            # If more than one site in input data, also create network-wide completeness figures (rows = pollutants, columns = time period,cell= % of network with complete data)
            if total_site_count > 1 :
                
                # Go back to this method if allowing for more varying network size
                # df_network = (pl.from_pandas(df).group_by([config_dict['timestamp_col'],config_dict['pollutant_col']])
                #     .agg(
                #         Site_Count = pl.col('Comp').sum(),
                #         Total_Site_Count = pl.col(config_dict['site_col']).count(),
                #     )).to_pandas().reset_index(drop=True)
                
                
                
                
                
            
                df_network = (df.groupby([config_dict['timestamp_col'],config_dict['pollutant_col']])['Comp'].sum()
                              .reset_index()
                              .rename(columns = {'Comp':'Site_Count'})
                              .merge(network_size,on=config_dict['pollutant_col']))
                
                
                

                df_network['Network_%'] = df_network['Site_Count']/df_network['Total_Site_Count']
        
                # Assign network completeness category based on %
                df_network['Network_Cat'] = df_network['Network_%'].apply(_network_category)
                
                
                #TODO-keep working here!
                
                # Group by years
                df_network['year'] = df_network[config_dict['timestamp_col']].dt.year
                
                if period=='Monthly':
                
                    df_network = df_network.rename(columns = {'year':'grouping'})
                
                # Group by 12 year chunks
                elif period=='Annual':
                    
                    df_network['grouping'] = (df_network["year"] - df_network["year"].min()) // 12 + 1
                
                for g in df_network['grouping'].unique():
                    
                    # Subset dataframe to group
                    df_n = df_network[df_network['grouping']==g]

                    # Get list of unique network completeness categories in dataframe (to subset color dictionary later)
                    cats = df_n['Network_Cat'].unique().astype(int).tolist()
                    cats.sort()
                    
                    #df_network['Network_%'] = round(df_network['Network_%']*100,0).astype(int).astype(str) + '%'
                    
                    # Add total site count per pollutant to pollutant column for plot y label
                    df_n['pollutant_site_count'] = df_n[config_dict['pollutant_col']] + ' (' + df_n['Total_Site_Count'].astype(str) + ' total sites)'
                    
                                    
                    # Create pivoted network dataframe where values are network categories for easier heatmap visualization
                    df_n_pv = df_n.pivot(index='pollutant_site_count',columns=config_dict['timestamp_col'],values='Network_Cat')
                    
                    df_n_pv.columns = [c.strftime(timestamp_format) for c in df_n_pv.columns]
                    
                    # Create subtitle describing range of data in figure
                    title_start = df_n_pv.columns[0]
                    title_end = df_n_pv.columns[-1]
                    
                    if title_start != title_end:
                        title_period = f'{title_start} to {title_end}'
                    else:
                        title_period = title_start
                    
                    #OLD method with variable network size
                    #df_network['Annot'] = df_network['Network_%'].astype(str) + '\n(' + df_network['Site_Count'].astype(str) + '/' + df_network['Network_Size'].astype(str) + ')'
    
                    # Created pivoted dataframe to annotate heatmap cells with actual %s
                    df_pv_annot = df_n.pivot(index='pollutant_site_count',columns=config_dict['timestamp_col'],values='Network_%')
                    
                    df_pv_annot.columns = [c.strftime(timestamp_format) for c in df_pv_annot.columns]
                    
                    fontsize = _fontsize(df_n_pv)
                    
                    # Create heatmap
                    fig,ax = plt.subplots(figsize=(_figsize(df_n_pv)))
                    plt.rcParams.update({"font.family" : "serif",'font.serif':['Arial']})
                    sns.heatmap(df_n_pv,cmap=[network_cat_colors[k] for k in cats],cbar=False,annot=df_pv_annot,
                                annot_kws={'size':(fontsize-2)},linewidths=0.25,linecolor='black',yticklabels=1,fmt='.0%')
                    plt.ylabel('')
                    plt.xlabel('')
                    plt.xticks(fontsize=fontsize,rotation=45)
                    plt.yticks(fontsize=fontsize)
        
                    # Create a custom legend defining complete and incomplete data colors
                    plt.legend(network_legend_shps, network_legend_txts,loc='center left',bbox_to_anchor=(1, 0.5),fontsize=fontsize)
        
                    ax.annotate(f'% of sites with complete data (completeness threshold: {comp_threshold}%). Based on number of unique sites per pollutant in data.',(0.2,0.2),fontsize=6,xycoords='figure points')
                    title = f'{period} Network-Wide Completeness'
                    plt.title(title + f'\n{title_period}',fontsize=fontsize)
                    plt.tight_layout()
                    plt.savefig(Path(output_file_path,title + f" {title_period.replace('/','_')}.png"),dpi=300)
                    plt.close(fig)
                
         
        # If period is Daily, review trends in network-wide daily data completeness for high variability, all sites/pollutants with 0% and make time series (if any exist or daily comp vis is true)
        else:
        
            # Compute network-wide daily data completeness - still usefule to show if only site to see variation over time
            df_network = (df.groupby([config_dict['timestamp_col'],config_dict['pollutant_col']])['Comp'].sum()
                          .reset_index()
                          .rename(columns = {'Comp':'Site_Count'})
                          .merge(network_size,on=config_dict['pollutant_col']))
            
            
            df_network['Network_%'] = round((df_network['Site_Count']/df_network['Total_Site_Count'])*100)
            
            # Check daily network completeness for pollutants with higher variability (completeness fluctuates across time - CV>=25%)
            cv_df = (df_network.groupby(config_dict['pollutant_col'])['Network_%'].std()/
             df_network.groupby(config_dict['pollutant_col'])['Network_%'].mean()).reset_index().rename(columns = {'Network_%':'Network_%_CV'})
            cv_check = cv_df[cv_df['Network_%_CV'] >= 0.25]
            
            print('Reviewing daily data for network-wide data completeness issues across dataset range (e.g., variable data completeness)...',end='\n\n')
            
            min_date = hr_df[config_dict['timestamp_col']].min().strftime('%Y-%m-%d')
            max_date = hr_df[config_dict['timestamp_col']].max().strftime('%Y-%m-%d')
            
            with open(summary_text_file,"a") as f:
                print(f'Reviewing daily data for network-wide data completeness issues across dataset range ({min_date} - {max_date}) (e.g., variable data completeness)...',end='\n\n',file=f)
            
            # Print the pollutants that have higher completeness variability
            if not cv_check.empty:
                
                with open(summary_text_file,"a") as f:
                
                    print("Daily network data completeness is noticeably variable for the following pollutants:",file=f)
                    
                    for __,row in cv_check.iterrows():
                
                         print(f"- {row[config_dict['pollutant_col']]} (CV : {round(row['Network_%_CV']*100)}%)",file=f)
                
                    print('Recommend reviewing this data for issues in device powering, data communication, or data processing to determine cause of data completeness variability.\nThis may also be due to natural changes in network size (e.g., adding more sites).',end='\n\n',file=f)
           
            # Check for days where no sites in network were complete per pollutant (0%)
            zero_check = df_network[df_network['Network_%']==0]
            
            # Print the pollutants that had periods of time with network-wide incomplete data and # of days when this occurred
            if not zero_check.empty:
                
                with open(summary_text_file,"a") as f:
                
                    print('There were network-wide data completeness issues (i.e., all sites at 0% for pollutant) for the following pollutants and cumulative days:',file=f)
                    
                    for pz in zero_check[config_dict['pollutant_col']].unique():
                      
                        z_day_count = zero_check[zero_check[config_dict['pollutant_col']]==pz].shape[0]
                        
                        print(f"- {pz} ({z_day_count} day(s))",file=f)
    
                    print('Recommend reviewing this data for issues in device powering, data communication, or data processing to determine cause of data completeness variability (e.g., weather impacts).',end='\n\n',file=f)

                # If multiple pollutants in dataset and some 0% in df, check for times when all pollutants had network-wide issues (sum of %'s = 0)
                if len(config_dict['pollutant_map'].keys())>1:
                    
                    all_sum = df_network.groupby(config_dict['timestamp_col'])['Network_%'].sum().reset_index()
                
                    zero_all_check = all_sum[all_sum['Network_%']==0]   
                
                    # Print the days where these network-wide pollutant-wide completeness issues occurred.
                    if not zero_all_check.empty:
                        
                        with open(summary_text_file,"a") as f:
                    
                            print('There were also network-wide data completeness issues across all pollutants on the following days:',file=f)
                            
                            print(zero_all_check[config_dict['timestamp_col']].dt.strftime(timestamp_format).tolist(),file=f)
            
            # If no issues found, print that.
            if cv_check.empty and zero_check.empty:
                
                with open(summary_text_file,"a") as f:
                
                    print('No issues found.',end='\n\n',file = f)
                    
            
            # If any suspect data completeness issues found OR daily_comp_vis was set True by user, output daily network completeness plot
            if daily_comp_vis or not cv_check.empty or not zero_check.empty:
                
                # Note: if site_count = 1, time series will just fluctuate between 0 and 100% but can still show useful trend.
                
                fig,ax = plt.subplots(figsize=(10,7))
                plt.rcParams.update({"font.family" : "serif",'font.serif':['Arial']})
                
                # Plot daily network data completeness per pollutant
                for p in config_dict['pollutant_map'].keys():
                    
                    # Define pollutant name as exists in df to subset appropriately in checks
                    p_name = config_dict['pollutant_map'][p]
                    dfp = df_network[df_network[config_dict['pollutant_col']] == p_name].reset_index(drop=True)
                    plt.plot(dfp[config_dict['timestamp_col']],dfp['Network_%'],label=p_name,linestyle='-',linewidth=2)
                 
                # Add shaded area at and above 75% - ideal threshold for network data completeness
                transf = transforms.blended_transform_factory(ax.transAxes,ax.transData)
                comp_patch = patches.Rectangle((0,comp_threshold),1,(100-comp_threshold),facecolor='forestgreen',alpha=0.2,transform=transf)
                ax.add_patch(comp_patch)
                
                plt.ylabel('Network Completeness (%)',fontsize=14)
                plt.yticks(fontsize=12)
                plt.ylim([-5,105]) #leave some room below and above realistic % limits
                plt.xticks(fontsize=10,rotation=45)
                
                # number of days to space x tick marks by to get 10 x axis tick marks
                n = ((df_network[config_dict['timestamp_col']].max() - df_network[config_dict['timestamp_col']].min()).days)/10 
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=round(n)))
                ax.xaxis.set_major_formatter(DateFormatter(timestamp_format))
                
                # Add annotation explaining what network % is based on
                ax.annotate('Network % based on number of unique sites per pollutant in input dataframe.',(0.3,0.5),fontsize=8,xycoords = 'axes points')
                
                # Add completenss patch to legend handles and labels
                handles,labels = ax.get_legend_handles_labels()
                handles.append(patches.Rectangle((0,0),1,1,facecolor='forestgreen',linewidth=0.5,alpha=0.4))
                labels.append('Ideal Network\nCompleteness Zone\n(>=75%)')
                
                # Increase # of legend columns dependent on # of legend items
                ncol = 1
                if len(labels)>3:
                    
                    ncol = 2
                
                plt.legend(handles=handles,labels=labels,fontsize=12,ncol=ncol)
                
                # Add title and save out
                title = f'{period} Network-Wide Completeness'
                plt.suptitle(title,fontsize=16,y=0.98)
                plt.title('(% of sites with complete data)',fontsize=12)
                plt.tight_layout()
                plt.savefig(Path(output_file_path,title + ' Time Series.png'),dpi=300)
                plt.close(fig)
                
    # Letting user know where output plots and txt file was saved
    print(f'All completeness plots and the "Data Completeness Evaluation Summary.txt" have been saved to {output_file_path}.',end='\n\n')
    
    # Status statement
    print('Data have undergone data completeness review....')
    
    # Prepare annual_df for output
    annual_df['Year'] = annual_df[config_dict['timestamp_col']].dt.year
    # Create bool Complete column from Comp (0/1)
    annual_df['Complete'] = annual_df['Comp']==1
    annual_df = annual_df.drop(['Comp',config_dict['timestamp_col']],axis=1).rename(columns = {config_dict['value_col']:'Annual_Average'})
    # Rearrange column order
    #other_cols = [c for c in hr_df.columns if c not in [config_dict['value_col'], config_dict['timestamp_col'], config_dict['site_col'],config_dict['lat_col'],config_dict['lon_col']]]
    #annual_df = annual_df[[config_dict['site_col'],other_cols,config_dict['lat_col'],config_dict['lon_col'],config_dict['pollutant_col'],'Year','Annual_Average','Complete']]
    
    return annual_df
