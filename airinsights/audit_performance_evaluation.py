#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:18:06 2026

"""
"""
@author: TD Enviro
"""


import pandas as pd
import numpy as np
from scipy import stats
import warnings
from geopy.distance import geodesic
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from airinsights.helpers import _validate_hourly
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import os
from airinsights.audit_helpers import _daily_comp


#-- Bias evalutation function
def _bias_eval(df,site_a,site_b):
    
    # Calculate r^2, slope, and offset. Dependent variable (sa - non-regulatory) and independent variable (sb - regulatory)
    res = stats.linregress(df[site_b], df[site_a])
    
    results = {'R^2':res.rvalue**2,'Slope':res.slope,'Offset':res.intercept}
    
    return results

#-- Precision evalutation function
def _prec_eval(df,site_a,site_b):
    
    # Negative needs to be removed for COD calculation (takes it out of 0-1 range)
    df[df<0] = np.nan
    
    # Remove any rows with NaN - days are not comparable
    df = df.dropna()
    
    num_days = len(df)
    
    if not df.empty:

        numerator = df[site_a] - df[site_b]
        denominator = df[site_a] + df[site_b]
        
        cod = sqrt(sum((numerator/denominator)**2)/num_days)
    
    # Fill in placeholder nan
    else:
        
        cod = np.nan
        
    results = {'COD':cod}
    
    
    return results, num_days

#-- Simple function for comparing dataframe results to performance metric criteria min and max
def _compare_to_metrics(df,metrics,result_col):
    
    # Needs df with Metric and Result column (long data format)
    df[['Min','Max']] = df['Metric'].map(metrics).tolist()
    df['Meets'] = np.where((df[result_col]>= df['Min']) & (df[result_col]<= df['Max']),True,False)
    
    return df




def audit_performance_evaluation(input_data : pd.DataFrame,
                    config : dict,
                    eval_comp_dist_km : int | float | None = 1.5, #TODO - confirm this default comparison distance with EDF
                    regulatory_sites : list | None = [],
                    output_file_path : str | None = os.getcwd()
                    ):
    """Comprehensive data quality evaluation based on 1) major data distribution review (hourly) and 
        2) site comparisons to inform bias and precision (daily).


    Parameters
    ----------
    input_data : pd.DataFrame
        A pandas DataFrame containing hourly AQ data (read in using read_aqdata_file()). Any regulatory data is expected to be in this input_data alongside sensor data.
    
    config : dict
        A dictionary detailing the input_data structure, containing input parameter names and values. 
    
    eval_comp_dist_km : dict, optional
        max evaluation distance for nearest neighbor comparisons(km). Default or user-defined. int or float
        Note: All possible site comparisons will be performed, but final network-level bias and precision guidance ('evaluation') will be based on data from comparisons within the provided or default comparison distance.
        If no site comparisons within the evaluation comparison distance are available, all comparison data will be used to inform bias and precision guidance with a warning to evaluate results with caution.
    
    regulatory_sites : list, optional
        List of regulatory sites to compare against other sites for bias calculations. If None, bias cannot be estimated.
        Regulatory data should be present alongside sensor data in input_data.
    
    output_file_path : str, optional
        The file path where output .txt files and figures should be saved. If None provided, current working directory will be used.
    
    #TODO - in future add monthly_evaluation bool, default False (would show median network r^2 and COD per month)
    
    
    Returns
    -------
    
    network_median_df : pd.DataFrame
        A pandas dataframe containing median network performance evaluation results per pollutant:
            
            **Pollutant**: Pollutant evaluated
            
            **Evaluation**: Bias or precision evaluation category (metric grouping)
            
            **Metric**: Metric calculated (R^2, offset, slope, or COD)
            
            **Median Result**: median result per metric across all evaluation site comparisons in network
            
            **Site Pair Count**: Number of unique evaluation site comparisons that went into the network median result
            
            **# of Days**: # of comparison days that went into the metric calculation for the network median result
            
            **Median Comparison Distance (km)**: median evaluation site comparison distance across data that informed the network median result
            
            **Min Comparison Distance (km)**: minimum evaluation site comparison distance across data that informed the network median result

            **Max Comparison Distance (km)**: maximum evaluation site comparison distance across data that informed the network median result
            
            **Meets**: Whether the network median result meets metric threshold criteria. For holistic bias and precision evaluation results, see Performance Evaluation Summary.txt

    Performance Evaluation Summary (.txt saved to output_file_path)
        Text summarizing bias and precision results with data use guidance
        Will also highlight any issue sites that were flagged during data distribution review or median site performance review
    
    Performance Evaluation Results visualized in seaborn strip plots per pollutant (.jpg's saved to output_file_path)
        Strip plot per metric with all site comparison results shown (evaluation comparisons are visually distinguished from the rest, where relevant)
    
    Median Site-Specific Evaluation Results table(.csv saved to output_file_path)
        Median evaluation results per site and pollutant (limited to evaluation comparison distance) to help investigate any issues sites highlighted in Performance Evaluation Summary
    
            **Pollutant**: Pollutant evaluated
           
            **Site**: site in input_data
            
            **Evaluation**: Bias or precision evaluation category (metric grouping)
            
            **Metric**: Metric calculated (R^2, offset, slope, or COD) 
            
            **Median Result**: Median result per metric across all evaluation site comparisons per site
            
            **Site Pair Count**: Number of unique evaluation site comparisons that went into the site median result
            
            **# of Days**: # of comparison days that went into the metric calculation for the site median result
            
            **Median Comparison Distance (km)**: median evaluation site comparison distance across data that informed the site median result

    
    Verbose Site Comparison Results table(.csv saved to output_file_path)
        A pandas DataFrame of bias and precision results per site comparison (all comparison distances).
            
            **Pollutant**: Pollutant evaluated
           
            **Site**: site in input_data (will not include regulatory, if present in function input)
            
            **Comparison Site**: site compared to (will include regulatory, if present in function input)
            
            **Evaluation**: Bias or precision evaluation category (metric grouping)
            
            **Metric**: Metric calculated (R^2, offset, slope, or COD) 
         
            **Result**: Result of metric calculation for site comparison
            
            **# of Days**: # of comparison days that went into the metric calculation for the site pair comparison
            
            **Distance (km)**: Distance between the site pair
    
        
    Notes
    -----


    """

    #-- Performance evaluation metric criteria
    metric_criteria = {'R^2':[0.7,1],'Slope':[0.65,1.35],'Offset':[-5,5],'COD':[0,0.45]}
    # Eval_categories dictionary contains standard text that will populate in the summary txt file to help with data use guidance
    eval_categories = {'Bias':{'metrics':['R^2','Slope','Offset'],'question':'Can the data be used to compare against regulatory health-based thresholds?'},
                       'Precision':{'metrics':['COD'],'question': 'Can data be compared across sites (e.g., spatial hot spot analysis)?'}}
    
    
    #-- Read in data, make sure contains hourly, and has at least 25 days in its time range (requirement for evaluation)
    df = input_data.copy()
    config_dict = config.copy()
    # Will raise an error if data only contains non-hourly freq or will print out how much data needed to be excluded
    df = _validate_hourly(df,config_dict)
    

    # Subset config_dict to pollutants that are actually in the dataset
    p_in_df = [pp for pp in config_dict['pollutant_map'].keys() if config_dict['pollutant_map'][pp] in df[config_dict['pollutant_col']].unique()]
    config_dict['pollutant_map'] = {k: config_dict['pollutant_map'][k] for k in p_in_df}
    

    #-- Check format of thresholds and other input fields and set default thresholds where not defined

    if not isinstance(eval_comp_dist_km, int) | isinstance(eval_comp_dist_km, float): 
        raise TypeError(f"integer or float expected for comparison distance, got {type(eval_comp_dist_km).__name__}.")
    
    elif eval_comp_dist_km >= 1000:
        
        # Set warning if comparison distance is very high (user may want to compare across entire large network, so just throw warning)
        warnings.warn(f'Comparison distance is set very high ({eval_comp_dist_km}km), check whether set in correct units (kilometers).')

    
    # Output_file_path
    if not isinstance(output_file_path, str): 
        raise TypeError(f"string expected, got {type(output_file_path).__name__}")
        
        
    # Convert comp_dist to float (if not already) and count decimal places to support distance comparisons later
    eval_comp_dist_km = float(eval_comp_dist_km)
    decimal_n = len(str(eval_comp_dist_km).split('.')[1]) # to help with rounding comparison distances
    
   
    # Unique sites in input data
    sites = df[config_dict['site_col']].unique()
    
    #-- Set performance_eval bool to inform further program sections
    performance_eval = True
    
    # Print warning if there is only one site in the input dataframe (cannot perform either precision or bias evaluation, but can still review data distribution)
    if len(sites)==1:
        
        warnings.warn('Only 1 site present in input data. Precision and bias site comparisons cannot be performed. Only data distribution evaluation will be performed.')
        
        performance_eval = False
        
    #-- Check regulatory site list contents
    if len(regulatory_sites)>0:
        
        bias_check_list = [[not isinstance(regulatory_sites, list),TypeError(f"List expected for regulatory_sites, got {type(regulatory_sites).__name__}")],
                            [len(set(regulatory_sites).intersection(set(sites))) == 0,ValueError(f"None of the user-specified regulatory_sites are present in input data {config_dict['site_col']} column.")],
                            [len(set(regulatory_sites).intersection(set(sites))) == len(sites),
                             ValueError('Input data contains only regulatory sites. Do not need to specify regulatory_sites in this case. Bias evaluation is based on comparison between available regulatory sites and "other" sites.')]
                             ]
        # Loop over checks, breaking and raising error if any are false
        for check in bias_check_list: 
            
            if check[0]:
                raise check[1]

    #-- Create location metadata for all sites
    sites_metadata = (df.drop_duplicates(subset = [config_dict['site_col'],config_dict['lon_col'],config_dict['lat_col']]).reset_index(drop=True))

    #-- Create text file to save out performance evaluation summary
    summary_text_file = Path(output_file_path,'Performance_Evaluation_Summary.txt')
    
    print('Data are undergoing performance evaluation...',end='\n\n')

    #-- First, identify sites with major data distribution issues (uses hourly data and 15th percentile, checking if less than or equal to zero - unitless)
    quantiles = df.groupby([config_dict['site_col'],config_dict['pollutant_col']])[config_dict['value_col']].quantile(.15).reset_index()
    quantiles = quantiles[quantiles[config_dict['value_col']]<=0]

    if not quantiles.empty:
        
        with open(summary_text_file,'w') as f:
        
            print('The following sites and pollutants were flagged during data distribution evaluation (15th percentile is less than or equal to 0):',file=f)
            for _,row in quantiles.iterrows():
                #TODO - currently rounding to 4th decimal place - check if this is ok?
                print(f" - Site {row[config_dict['site_col']]} - {row[config_dict['pollutant_col']]} (15th percentile = {round(row[config_dict['value_col']],4)})",file=f)
            print('This could be caused by a couple of issues: calibration data included in dataset, data below minimum detection limit (MDL) is being set to zero, baseline data issues, etc.',file=f)
            print('Recommend reviewing these sites and pollutants for these issues.',file=f,end='\n\n')
        
        print('Issue sites detected during data distribution evaluation and saved to Performance Evaluation Summary.',end='\n\n')
        
    else:
        with open(summary_text_file,'w') as f:
            print('No issue sites detected through data distribution evaluation.',file=f,end='\n\n')
    
    
    #-- Perform bias and precision evaluation, comparing sites across the dataset
    if performance_eval:
        # Average data to daily for bias and precision evaluation (smoothes out data so that we're not focused on intermittent real local pollution differences)
        day_df = _daily_comp(df,config_dict,75)
        
        # Create result list to add to throughout for loop
        full_results = []
            
        #-- Loop through pollutant_map in config_dict 
        for p in config_dict['pollutant_map'].keys():
             
            # Define pollutant name as exists in input data and subset day_df
            p_name = config_dict['pollutant_map'][p]
            pdf = day_df[day_df[config_dict['pollutant_col']]==p_name]
    
            # Define what sites are present per pollutant (p_all_sites and p_sites will be the same if no regulatory sites)
            p_all_sites = pdf[config_dict['site_col']].unique()
            p_sites = [ps for ps in p_all_sites if ps not in regulatory_sites]
    
            #-- Loop through sites per pollutant (p_sites = all non-regulatory sites, Site A = all sites, Site B = all sites + regulatory)
            for sa in p_sites:
                
                sa_latlon = [sites_metadata[sites_metadata[config_dict['site_col']]==sa][config_dict['lat_col']].values[0],sites_metadata[sites_metadata[config_dict['site_col']]==sa][config_dict['lon_col']].values[0]]
                
                # Loop through all sites (Site B), including regulatory (these won't get compared to twice since not in p_sites above)
                for sb in p_all_sites: 
                
                    if sb != sa:
                    
                        sb_latlon= [sites_metadata[sites_metadata[config_dict['site_col']]==sb][config_dict['lat_col']].values[0],sites_metadata[sites_metadata[config_dict['site_col']]==sb][config_dict['lon_col']].values[0]]
    
                        # Calculate distance between sites
                        distance = round(geodesic(sa_latlon, sb_latlon).km,decimal_n)
                        
                        # Subset dataframe to two sites
                        p_compare_df = pdf[pdf[config_dict['site_col']].isin([sa,sb])]
                        
                        # Pivot for easier comparison
                        p_compare_df = pd.pivot(p_compare_df,index = config_dict['timestamp_col'],columns = config_dict['site_col'], values = config_dict['value_col'])
                        
                        # Remove any rows with NaN (these days are not comparable)
                        p_compare_df = p_compare_df.dropna()
                        
                        # If comparison dataset isn't empty 
                        if not p_compare_df.empty:
    
                            #-- Perform bias evaluation
                            if sb in regulatory_sites:
                                
                                # Number of comparison days
                                ns = len(p_compare_df)
                                
                                # Run bias evaluation internal function
                                results = _bias_eval(p_compare_df,sa,sb)
                                
                                eval_t = 'Bias'
        
                            #-- Perform precision evaluation    
                            else:
                                
                                # Run precision evaluation internal function (and get number of days after removing negative data for COD calculation)
                                results, ns = _prec_eval(p_compare_df,sa,sb)
                                
                                eval_t = 'Precision'
                            
    
                            #-- Append all site comparison results to full_results list
                            
                            # Loop through metric results from bias or prec functions
                            for mt in results.keys():
    
                                mr = results[mt]
                                
                                full_results.append({'Standard Pollutant':p,'Site A':sa,'Site B':sb,'Evaluation':eval_t,'Metric':mt,'Result':mr,'# of Days':ns,'Distance (km)':distance})
                    
       
        #-- Process bias and precision evaluation results
        
        # Convert list of results into dataframe
        full_result_df = pd.DataFrame(full_results)
        
        # Round results to two decimal places
        full_result_df['Result'] = round(full_result_df['Result'],2)
        
        # Remove rows where Result is nan (placeholder in precision calc if all negative data)
        full_result_df = full_result_df[~full_result_df['Result'].isna()]
        
        
        # Check whether site comparisons are in the evaluation comparison distance per pollutant and evaluation method.
        eval_dist_check = ((full_result_df.groupby(['Standard Pollutant','Evaluation'])['Distance (km)'].min())
                           .reset_index()
                           .loc[lambda x: x['Distance (km)'] <= eval_comp_dist_km])
                            
        # Create Eval Type bool column that distinguishes between data that should be used for evaluation or not
        sub_result_df = full_result_df.copy()
        sub_result_df['Eval_Type'] = ''
        for _, row in eval_dist_check.iterrows():
            
            sub_result_df['Eval_Type'] = np.where(((sub_result_df['Standard Pollutant']==row['Standard Pollutant']) 
                              & (sub_result_df['Evaluation']==row['Evaluation'])
                              & (sub_result_df['Distance (km)'] > eval_comp_dist_km)),False,sub_result_df['Eval_Type'])
            
        sub_result_df.loc[sub_result_df['Eval_Type'] != False,'Eval_Type'] = True
                              
        # Create site median results (subset to evaluation comparison distance, Eval_Type = True)
        site_median_df = (sub_result_df[sub_result_df['Eval_Type']]
                         .groupby(['Standard Pollutant','Site A','Metric'])
                         .agg(**{'Median Result': ('Result', 'median'),
                                 'Site Pair Count': ('Result', 'count'),
                                 '# of Days': ('# of Days', 'sum'),
                                 'Median Comparison Distance (km)': ('Distance (km)','median')})
                         .reset_index())
        site_median_df['Median Result'] = round(site_median_df['Median Result'],2)
        site_median_df['Median Comparison Distance (km)'] = round(site_median_df['Median Comparison Distance (km)'],decimal_n)
        
        # Remove duplicate site pairs from sub_result_tbl (occur during precision comparison - but don't want to keep these going forward to not inflate network site pair count per metric)
        sub_result_df['Site Pair'] = sub_result_df[['Site A','Site B']].values.tolist()
        sub_result_df['Site Pair'] = sub_result_df['Site Pair'].apply(lambda x: ''.join(sorted(x)))
        sub_result_df = sub_result_df.drop_duplicates(subset = ['Standard Pollutant','Metric','Site Pair'])
        
        
        #-- Calculate network median metric per pollutant and compare against metric criteria thresholds (include comparison distance ranges in df) (subset to evaluation comparison distance, Eval_Type = True)
        network_median_df = (sub_result_df[sub_result_df['Eval_Type']]
                             .groupby(['Standard Pollutant','Metric'])
                             .agg(**{'Median Result': ('Result', 'median'),
                                      'Site Pair Count': ('Result', 'count'),
                                      '# of Days' : ('# of Days', 'sum'),
                                      'Median Comparison Distance (km)': ('Distance (km)','median'),
                                      'Min Comparison Distance (km)': ('Distance (km)','min'),
                                      'Max Comparison Distance (km)': ('Distance (km)','max')})
                              .reset_index()
                              .pipe(_compare_to_metrics,metric_criteria,'Median Result')
                              .drop(['Min','Max'],axis=1))
        network_median_df['Median Result'] = round(network_median_df['Median Result'],2)
        network_median_df['Median Comparison Distance (km)'] = round(network_median_df['Median Comparison Distance (km)'],decimal_n)
                              
            
        # Identify any pollutants where there are fewer than 3 site comparisons for precision criteria (for later warning)
        warning_prec_polls = (network_median_df[(network_median_df['Site Pair Count'] < 3)
                              & (network_median_df['Metric'].isin(eval_categories['Precision']['metrics']))]
                              ['Standard Pollutant'].unique().tolist())
    
        performance_summary = []
        
        # Remind user what evaluation comparison distance was used with note before showing performance evaluation results and guidance
        with open(summary_text_file,"a") as f:
            print(f"Evaluation comparison distance set at {eval_comp_dist_km}km.\nNote: Evaluations are based on available data, and actual evaluation comparison distances may be greater or smaller. These are listed per evaluation below.",file=f,end='\n\n')
        
        #-- Loop through performance evaluation categories and describe results to user in summary text file
        
        for cat in eval_categories.keys():
            
            # Subset network median dataframe to category
            nm_df = network_median_df[network_median_df['Metric'].isin(eval_categories[cat]['metrics'])]
            
            # Identify any pollutants that are not in this eval category but in config_dict (for later warning)
            missing_polls = [p for p in config_dict['pollutant_map'].keys() if p not in nm_df['Standard Pollutant'].unique()]
            
            # Identify any pollutants where minimum comparisons distance is greater than evaluation comparison distance (for later warning)
            warning_polls = nm_df[nm_df['Min Comparison Distance (km)'] > eval_comp_dist_km]['Standard Pollutant'].unique().tolist()
            
            # Identify any pollutants where the # of comparison days is less than 2 weeks (for later warning)
            warning_time_polls = nm_df[nm_df['# of Days'] < 14]['Standard Pollutant'].unique().tolist()
            
            # Begin writing to summary text file
            with open(summary_text_file,"a") as f:
                print(f"{cat} evaluation, answering '{eval_categories[cat]['question']}':",file=f)
                
                # If there is any data for this evaluation category, loop through pollutants and print data use guidance along with any warning text
                if not nm_df.empty:
                    
                    # Print warning text, if any warning_polls
                    if len(warning_polls)>0:
                        print(f"WARNING: Review {cat} guidance for {', '.join(warning_polls)} with caution. No site comparisons are within evaluation comparison distance ({eval_comp_dist_km} km).",file=f)
                    
                    if len(warning_time_polls)>0:
                        print(f"WARNING: Review {cat} guidance for {', '.join(warning_polls)} with caution. Network evaluation based on less than 2 weeks of data.",file=f)
                    
                    if (len(warning_prec_polls)>0) & (cat=='Precision'):
                        print(f"WARNING: Review {cat} guidance for {', '.join(warning_prec_polls)} with caution. Network evaluation based on less than 3 site comparisons.")
                    
                    for i in range(0,len(nm_df['Standard Pollutant'].unique())):
                        
                        p = nm_df['Standard Pollutant'].unique()[i]
                        
                        p_name = config_dict['pollutant_map'][p]
                        
                        pnm_df = nm_df[nm_df['Standard Pollutant']==p]
                        
                        med_dist = pnm_df['Median Comparison Distance (km)'].unique()[0]
                        
                        min_dist = pnm_df['Min Comparison Distance (km)'].unique()[0]
                        
                        max_dist = pnm_df['Max Comparison Distance (km)'].unique()[0]
                        
                        #-- For bias, three possible scenarios: all criteria met, r^2 met but either slope and/or offset not met, r^2 not met
                        if cat == 'Bias':
                            
                            # Pivot dataframe for easier comparison with values equal to boolean from _compare_to_metrics
                            pnm_df = pnm_df.pivot(index='Standard Pollutant',columns='Metric',values='Meets')
                            
                            # If r^2 met
                            if pnm_df['R^2'].item():
                                
                                # Check if all other criteria met
                                if (pnm_df[['Slope','Offset']]).all().all():
                                    print(f"  {p_name} - Yes, data are accurate (all criteria met ({', '.join(eval_categories[cat]['metrics'])}) with median {med_dist}km eval comparison distance ({min_dist} - {max_dist}km range)).",file = f)
                                    
                                    performance_summary.append({'Pollutant':p_name,'Evaluation':cat,'Pass':True})
                                
                                # Slope and/or offset not met
                                else:
                                    print(f'  {p_name} - Maybe, but data correction may be needed (R^2 criteria met, but offset and/or slope criteria outside of ideal range with median {med_dist}km eval comparison distance ({min_dist} - {max_dist}km range)).',file = f)
                                    
                                    performance_summary.append({'Pollutant':p_name,'Evaluation':cat,'Pass':True})
    
                            # R^2 not met
                            else:
                                print(f"  {p_name} - No, data are not accurate (most important criteria, R^2, was not met with median {med_dist}km eval comparison distance ({min_dist} - {max_dist}km range)).",file = f)
                                
                                performance_summary.append({'Pollutant':p_name,'Evaluation':cat,'Pass':False})
                                                        
                            
                        if cat == 'Precision':
                            
                            # (just one metric to check - COD, so don't need to pivot like for bias)
                            
                            if pnm_df['Meets'].item():
                                
                                if (pnm_df['Median Result'] <= 0.3).item():
                                    print(f"  {p_name} - Yes, data are precise/comparable (COD<=0.3 with median {med_dist}km eval comparison distance ({min_dist} - {max_dist}km range)).",file = f)
                                    
                                    performance_summary.append({'Pollutant':p_name,'Evaluation':cat,'Pass':True})
    
                                    
                                # Else, COD is greater than 0.3 but less than the max range (0.45)
                                else:
                                    print(f"  {p_name} - Maybe, but some divergence found on median across sites (0.3  < COD <= 0.45 with median {med_dist}km eval comparison distance ({min_dist} - {max_dist}km range)). Could be real or a sign of imprecise data.",file = f)
                                    
                                    performance_summary.append({'Pollutant':p_name,'Evaluation':cat,'Pass':True})
    
                                
                            # Else, COD is greater than or equal to max range (0.45)
                            else:
                                print(f"  {p_name} - No, data are not precise (COD>={metric_criteria['COD'][1]} with median {med_dist}km eval comparison distance ({min_dist} - {max_dist}km range)).",file = f)
                                
                                performance_summary.append({'Pollutant':p_name,'Evaluation':cat,'Pass':False})
                    
                    # Add text about any pollutants missing data for these evaluations (outside of pollutant for loop)
                    if len(missing_polls)>0:
                        
                        print(f"  {', '.join(missing_polls)} - Could not perform {cat} evaluation (no site comparisons or data available).",file=f)
    
                    # Add text letting user know where to find verbose results
                    print("Note: Individual site performance results (regardless of site comparison distance) can be reviewed in 'Verbose Site Comparison Results.csv'",file=f)
    
                    
                # If no data in this evaluation category, let user know to look in console output for more info
                else:
                    print(f'  {cat} evaluation could not be performed (no site comparisons or data available).',file=f)
                
                # Add line break between bias and precision results for text organization     
                print('\n',file=f)
        
        
        #-- Make strip plots summarizing all metric data (one output figure per pollutant)
        
        for p in sub_result_df['Standard Pollutant'].unique():
            
            # Define what metrics were calculated per pollutant
            metrics = sub_result_df.loc[sub_result_df['Standard Pollutant'] == p,'Metric'].unique()
            
            p_name = config_dict['pollutant_map'][p]
            
    
            #-- Set up figure dimensions (dependent on number of subplots)
            # Number of metrics = number of subplots
            plot_n = len(metrics)
            # Fixed subplot width (inches)
            plot_w = 3
            # Fixed margins for legend (inches)
            left_in, right_in = 1.0, 2.0
            # Figure width based on plot size and margins (inches)
            fig_w = (plot_n * plot_w) + left_in + right_in
            
            # Start counting subplot index starting at 1
            iii = 1
            fig = plt.figure(figsize=(fig_w,7))
            plt.rcParams.update({"font.family" : "serif",'font.serif':['Arial']})
    
            # Loop through metrics and add strip plot to figure - use full_result_tbl
            for m in metrics:
                
                # Site comparison data points that went into network-level evaluation
                df_plot = sub_result_df[(sub_result_df['Metric'] == m) & (sub_result_df['Standard Pollutant'] == p)]
                
                df_plot['Eval_Type'] = np.where(df_plot['Eval_Type']==True,'Eval','Other')
                
                # Define min and max of ideal metric range
                m_min, m_max = metric_criteria[m]
                
                #-- Define min and max for y axis depending on metric
                # R^2 and COD have a set possible range of values
                if (m == 'R^2') | (m == 'COD') :
                    min_y = 0
                    max_y = 1
                
                # Slope and offset have an undefined range, base on data/thresholds
                else:
                    min_y = min(m_min,min(df_plot['Result'])) - 0.5
                    max_y = max(m_max,max(df_plot['Result'])) + 0.5
                
                # Set up subplots based on number of metrics
                ax = plt.subplot(1,plot_n,iii)
                
                
                # Plot data and configure subplots
                sns.stripplot(df_plot,hue='Eval_Type',y='Result',size=10,palette = {'Eval':'cornflowerblue','Other':'grey'},edgecolor='white',linewidth=1,legend=False,dodge=True)
                plt.ylabel(m,fontsize=16,weight='bold')
                plt.yticks(fontsize=15)
                plt.ylim([min_y,max_y])
                ax.set_xlim([-1,1])
                plt.legend([])
                plt.xticks([])
                
                # Add green shading to indicate ideal range for each metric
                rect = Rectangle((-1,m_min), 2, (m_max-m_min), color='forestgreen',alpha=0.1)
                ax.add_patch(rect)
                
                # Include network median results on plot as a horizontal line and comparison distance median and range
                n_tbl = network_median_df[(network_median_df['Metric'] == m) & (network_median_df['Standard Pollutant']==p)]
                for _,row in n_tbl.iterrows():
                    ax.hlines(row['Median Result'],color='cornflowerblue',linestyle='dashed',linewidth=2,xmin=-1,xmax=1)
                
                # Add to subplot count
                iii+=1
            
            #-- Create a custom legend to describe each plot element
            custom_legend = []
            custom_legend.append(patches.Rectangle((0,0),1,1,facecolor='forestgreen',linewidth=0.5,alpha=0.4))
            custom_legend.append(Line2D([0], [0],color='cornflowerblue', lw=3,linestyle='dashed'))
            custom_legend.append(Line2D([0], [0], marker='o', color='cornflowerblue',linestyle='', markersize=10))
            custom_labels = ['Ideal\nResult Range','Network\nMedian','Site Comparisons\n(in network eval)']
            
            # Include legend marker and label for other site comparisons if present in data
            if df_plot['Eval_Type'].nunique()>1:
                custom_legend.append(Line2D([0], [0], marker='o', color='grey',linestyle='', markersize=10))
                custom_labels.append('Site Comparisons\n(outside network eval)')
            
            plt.legend(handles=custom_legend,labels=custom_labels,fontsize=12)
            sns.move_legend(ax,"upper left",bbox_to_anchor=(1,1))
            
            # Add comparison distance as text to bottom of plot and description of what metrics inform bias and precision
            sub_text = ''
            if 'R^2' in metrics:
                n_tbl = network_median_df[(network_median_df['Metric'] == 'R^2') & (network_median_df['Standard Pollutant']==p)]
                med_dist = n_tbl['Median Comparison Distance (km)'].unique()[0]
                min_dist = n_tbl['Min Comparison Distance (km)'].unique()[0]
                max_dist = n_tbl['Max Comparison Distance (km)'].unique()[0]
                sub_text += f'\nBias: R^2, Slope, and Offset (comparison to regulatory site(s), {med_dist}km median eval distance).'
            if 'COD' in metrics:
                n_tbl = network_median_df[(network_median_df['Metric'] == 'COD') & (network_median_df['Standard Pollutant']==p)]
                med_dist = n_tbl['Median Comparison Distance (km)'].unique()[0]
                min_dist = n_tbl['Min Comparison Distance (km)'].unique()[0]
                max_dist = n_tbl['Max Comparison Distance (km)'].unique()[0]
                sub_text += f'\nPrecision: COD (comparison to other site(s), {med_dist}km median eval distance).'
            
            # Make more space on the right for the legend (move figures around), and at the bottom for the text, and remove lefthand white space
            fig.subplots_adjust(bottom=0.15,wspace=0.4,left=(left_in / fig_w),right = (1-right_in/fig_w)) 
            
            # Set up figure title and layout depending on how many subplots there are
            if plot_n>1:
                title = f'Performance Evaluation Results for {p_name}'
                plt.suptitle(title,fontsize=16)
                fig.text(fig.axes[0].get_position().x0,0.03,sub_text,fontsize=10)
    
            else:
                title = f'Performance Evaluation Results\nfor {p_name}'
                plt.title(title,fontsize=16)
                fig.text(0.01,0.03,sub_text,fontsize=10)
    
            
            # Save out figure to output file path
            plt.savefig(Path(output_file_path,f'Performance Evaluation Results for {p_name}.png'),dpi=300)
            plt.close(fig)    
        
        
        #-- Save out performance evaluation dataframes
            
        # Create metric to evaluation mapping to add to output dataframes (already in full_result_df)
        metrics_dict = {m: cat for cat in eval_categories.keys() for m in eval_categories[cat]['metrics']}  
        
        # Rename columns, add eval categories, and map standard pollutants to config pollutant names
        network_median_df.insert(0,'Pollutant',network_median_df['Standard Pollutant'].map(lambda pn: config_dict['pollutant_map'][pn]))
        network_median_df.insert(2,'Evaluation',network_median_df['Metric'].map(metrics_dict))
        network_median_df = network_median_df.drop('Standard Pollutant',axis=1)
        
        # Rename columns and map standard pollutants to config pollutant names and save out
        full_result_df.insert(0,'Pollutant', full_result_df['Standard Pollutant'].map(lambda pn: config_dict['pollutant_map'][pn]))
        full_result_df = full_result_df.rename(columns = {'Site B':'Comparison Site','Site A':'Site'}).drop('Standard Pollutant',axis=1)
        full_result_df.to_csv(Path(output_file_path,'Verbose Site Comparison Results.csv'),index=False)
        
        
        # Rename columns, add eval categories, and map standard pollutants to config pollutant names and save out
        site_median_df.insert(0,'Pollutant',site_median_df['Standard Pollutant'].map(lambda pn: config_dict['pollutant_map'][pn]))
        site_median_df.insert(3,'Evaluation',site_median_df['Metric'].map(metrics_dict))
        site_median_df = site_median_df.rename(columns = {'Site A':'Site'}).drop('Standard Pollutant',axis=1)
        site_median_df.to_csv(Path(output_file_path,'Median Site-Specific Evaluation Results.csv'),index=False)
        
        
        #-- Identify any median issue sites
        
        # Identify any sites per pollutant where network-level evaluation passed but site-median levels don't meet criteria (focused on R^2 or COD)
        issue_sites = (site_median_df.pipe(_compare_to_metrics,metric_criteria,'Median Result')
                       .merge(pd.DataFrame(performance_summary),on=['Pollutant','Evaluation'])
                       .loc[lambda x: x['Metric'].isin(['R^2','COD']) & ~x['Meets'] & x['Pass']]
                       .groupby(['Site','Pollutant'])['Metric'].apply(list)
                       .reset_index())
        
        # List out issue sites in summary text file
        with open(summary_text_file,"a") as f:
            
            if not issue_sites.empty:
                print('The following sites do not meet correlation and/or precision criteria (R^2 and COD) while network-level criteria are met. Recommend review - there may be issues with siting, instrumentation, data processing, or influence from real local sources.',file=f)
                
                for _,row in issue_sites.iterrows():
                    print(f"  {row['Pollutant']} at {row['Site']} ({', '.join(row['Metric'])})",file=f)
                
                print('Review further detail in Median Site-Specific Performance Results.csv',file=f)
                        
            else:
                print('No individual issue sites found while reviewing correlation and precision criteria (R^2 and COD) for pollutants that meet network-level criteria.',file=f)
                
                
        #TODO - future version: calculate metrics per month and show how they change over time

        
        # Let user know where output plots and txt file was saved
        print(f'The "Performance Evaluation Summary.txt" and any plots and files have been saved to {output_file_path}.',end='\n\n')
        return network_median_df
    
    else:
        
        print(f'The "Performance Evaluation Summary.txt" has been saved to {output_file_path}.',end='\n\n')
        return (pd.DataFrame(columns = ['Pollutant', 'Evaluation', 'Metric', 'Median Result', 'Site Pair Count',
               '# of Days', 'Median Comparison Distance (km)',
               'Min Comparison Distance (km)', 'Max Comparison Distance (km)',
               'Meets']))
        
         
        
    


