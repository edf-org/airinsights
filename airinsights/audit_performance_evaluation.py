#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:18:06 2026

"""
"""
@author: TD Enviro
"""


import pandas as pd
import geopandas as gpd
import numpy as np
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
from airinsights.internal_data_completeness_functions import _daily_comp





#-- Simple function for comparing dataframe results to performance metric criteria min and max
def _compare_to_metrics(df,metrics,result_col):
    
    # Needs df with Metric and Result column (long data format)
    df[['Min','Max']] = df['Metric'].map(metrics).tolist()
    df['Meets'] = np.where((df[result_col]>= df['Min']) & (df[result_col]<= df['Max']),True,False)
    
    return df




def performance_evaluation(input_data : pd.DataFrame,
                    config_dict : str,
                    comp_dist_km : dict | None = None, 
                    regulatory_sites : list | None = [],
                    verbose_bias_df : bool = False, 
                    output_file_path : str | None = os.getcwd()
                    ):
    """Comprehensive data quality evaluation based on 1) major hourly data distribution review and 
        2) nearest neighbor comparisons to inform bias and precision.


    Parameters
    ----------
    input_data : pd.DataFrame
        A pandas DataFrame containing hourly AQ data (read in using read_aqdata_file())
    
    config_dict : dict
        A dictionary detailing the input_data structure, containing input parameter names and values. 
    
    comp_dist_km : dict, optional
        max distance for nearest neighbor comparisons(km) per pollutant. Default or user-defined. {str: int or float}
        If only some pollutants specified in user-defined thresholds, program will try to fill in defaults.
        Default comp_dist_km: {'PM2_5':3,'O3':3,'PM10':1.5,'NO2':1.5,'CO':1.5,'SO2':1.5}
    
    regulatory_sites : list, optional
        List of regulatory sites to compare against other sites for bias calculations. If None, bias cannot be estimated.
        
    verbose_bias_df : bool, default False
        If True, output also includes a pandas DataFrame of bias evaluation results, showing performance results per regulatory site comparison, along with # of comparison days and distance (km)
    
    output_file_path : str, optional
        The file path where output .txt files and figures should be saved. If None provided, current working directory will be used.
    
    #TODO - in future add monthly_evaluation bool, default False (would show median network r^2 and COD per month)
    
    
    Returns
    -------
    
    network_median_tbl : pd.DataFrame
        A pandas dataframe containing median network performance evaluation results per pollutant:
            
            **Pollutant**: pollutant in input_data
            
            **Metric**: performance metric calculated (R^2, offset, slope, or COD)
            
            **Median Result**: median result per metric across all site comparisons in network
            
            **Site Pair Count**: the number of comparison site pairs that went into the network median result
            
    
    Performance Evaluation Summary (.txt saved to output_file_path)
        Text summarizing bias and precision results with data use guidance
        Will also highlight any issue sites that were flagged during data distribution review or median site performance review
    
    Performance Evaluation Results visualized in seaborn strip plots per pollutant (.jpg's saved to output_file_path)
    
    Site Median Performance Results table (.csv saved to output_file_path)
        Median performance results per site and pollutant to help investigate any issues sites highlighted in Performance Evaluation Summary
    
            **Pollutant**: pollutant in input_data
           
            **Site**: site in input_data
            
            **Metric**: metric calculated
            
            **Median Result**: median result per metric calculated
    
            
    verbose_bias_df : pd.DataFrame (.csv saved to output_file_path)
        A pandas DataFrame of summarized bias results per regulatory site comparison. Columns = Pollutant, Site, Regulatory Site, R^2, Slope, Offset,
            # of Comparison Days, Distance (km).
            
            **Pollutant**: pollutant in input_data
           
            **Site**: site in input_data (non regulatory site)
            
            **Regulatory Site**: regulatory site compared to
            
            **R^2**: R^2 from linear regression per site-regulatory site pair
            
            **Slope**: Slope from linear regression per site-regulatory site pair
         
            **Offset**: Offset from linear regression per site-regulatory site pair
            
            **# of Comparison Days**: R^2 per site-regulatory site pair
            
            **Distance (km)**: Distance between each site-regulatory site pair
    
        
    Notes
    -----


    """
    
    #-- Default comparison settings
    default_comp_dist_km = {'PM2_5':3,'O3':3,'PM10':1.5,'NO2':1.5,'CO':1.5,'SO2':1.5}
    generic_comp_dist_km = 3 

    #-- Performance evaluation metric criteria
    metric_criteria = {'R^2':[0.7,1],'Slope':[0.65,1.35],'Offset':[-5,5],'COD':[0,0.45]}
    # Eval_categories dictionary contains standard text that will populate in the summary txt file to help with data use guidance
    eval_categories = {'Bias':{'metrics':['R^2','Slope','Offset'],'question':'Can the data be used to compare against regulatory health-based thresholds?'},
                       'Precision':{'metrics':['COD'],'question': 'Can data be compared across sites (e.g., spatial hot spot analysis)?'}}
    
    
    #-- Read in data, make sure contains hourly, and has at least 25 days in its time range (requirement for evaluation)
    df = input_data.copy()
    # Will raise an error if data only contains non-hourly freq or will print out how much data needed to be excluded
    df = _validate_hourly(df,config_dict)
    
    df_day_count = (df[config_dict['timestamp_col']].max() - df[config_dict['timestamp_col']].min()).days
    if df_day_count < 25:
        raise ValueError(f'Dataset does not contain enough days for evaluation (at least 25 days required, dataset has {df_day_count}.')

    # Subset config_dict to pollutants that are actually in the dataset
    p_in_df = [pp for pp in config_dict['pollutant_map'].keys() if config_dict['pollutant_map'][pp] in df[config_dict['pollutant_col']].unique()]
    config_dict['pollutant_map'] = {k: config_dict['pollutant_map'][k] for k in p_in_df}
    

    #-- Check format of thresholds and other input fields and set default thresholds where not defined
    
    # Comparison distance
    if comp_dist_km is None:

        # Subset default_comp_dist_km dictionary to only pollutants in config_dict (helps with early on checks for any nearest neighbor sites)
        comp_dist_km = {c: default_comp_dist_km[c] for c in config_dict['pollutant_map'].keys()}
        
    if not isinstance(comp_dist_km, dict): 
        raise TypeError(f"dictionary expected for comp_dist_km, got {type(comp_dist_km).__name__}")
    
    # Output_file_path
    if not isinstance(output_file_path, str): 
        raise TypeError(f"string expected, got {type(output_file_path).__name__}")
       

    # Unique sites in input data
    sites = df[config_dict['site_col']].unique()
  
    
    #-- Check if can preform bias evaluation (initial check) 
    if len(regulatory_sites)>0:
        
        if not isinstance(regulatory_sites, list): 
            raise TypeError(f"List expected for regulatory_sites, got {type(regulatory_sites).__name__}")
            
        elif len(set(regulatory_sites).intersection(set(sites))) == 0:
            warnings.warn("None of the user-specified regulatory_sites are present in input data {config_dict['site_col']} column.\nBias evaluation will not be performed.")
            
            bias_evaluation = False
        
        elif len(set(regulatory_sites).intersection(set(sites))) == len(sites):
            warnings.warn('Input data contains only regulatory sites. Bias evaluation is based on comparison between available regulatory sites and "other" sites.\nBias evaluation will not be performed.')
            
            bias_evaluation = False
            
        else:
            
            bias_evaluation = True
    else:
        
        print("No regulatory sites provided for comparison. Bias evaluation will not be performed.")

        bias_evaluation = False
    
    
    #-- Create site list and geodataframe that is separate from regulatory sites (will be the same if regulatory_sites is an empty list)
    sites = [s for s in sites if s not in regulatory_sites]
    
    #-- Create location metadata for sites (not including regulatory_sites)
    sites_metadata = (df[df[config_dict['site_col']].isin(sites)]
                           .drop_duplicates(subset = [config_dict['site_col'],config_dict['lon_col'],config_dict['lat_col']]).reset_index(drop=True))
    gdf = gpd.GeoDataFrame(sites_metadata,geometry=gpd.points_from_xy(sites_metadata[config_dict['lon_col']], sites_metadata[config_dict['lat_col']]),crs="EPSG:4326")
    # Convert to UTM (projected) - sindex.nearest will then output in meters below
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    #TODO - discuss if assumption of taking first lat lon instance is ok - assumes lat,lon is static per site  
    
    #-- Check if can preform precision evaluation (i.e., if any sites within comparable distance of each other) 
    nearest_idx, nearest_m = gdf.sindex.nearest(gdf.geometry, exclusive=True, return_all=False, return_distance=True)
    gdf['nearest_m'] = np.round((nearest_m / 1000),3)
    # Check against max distance in comparison distance dictionary
    if (gdf['nearest_m'] > max(comp_dist_km.values())).all():
        
        print(f'No sites in comparison range. Precision evaluation will not be performed. Smallest distance between sites is {min(nearest_m)}km.')
        
        prec_evaluation = False
        
    else:
        
        prec_evaluation = True
    
    #-- Check if can actually perform bias evaluation (i.e., if any in comparable distance)
    if bias_evaluation:
        
        # Create separate reg metadata if regulatory_sites are appropriate for bias evaluation
        reg_metadata = (df[df[config_dict['site_col']].isin(regulatory_sites)]
                               .drop_duplicates(subset = [config_dict['site_col'],config_dict['lon_col'],config_dict['lat_col']]).reset_index(drop=True))
        
        reg_gdf = gpd.GeoDataFrame(reg_metadata,geometry=gpd.points_from_xy(reg_metadata[config_dict['lon_col']], reg_metadata[config_dict['lat_col']]),crs="EPSG:4326")
        # Convert to UTM (projected) - sindex.nearest will then output in meters below
        reg_gdf = reg_gdf.to_crs(reg_gdf.estimate_utm_crs())

        # Check if any non-regulatory sites in comparable distance to regulatory_sites (outputs nearest distance in meters)
        reg_nearest_idx, reg_nearest_m = reg_gdf.sindex.nearest(gdf.geometry, exclusive=True, return_all=False, return_distance=True)
        
        # Check against max distance in comparison distance dictionary
        if ((reg_nearest_m/1000) > max(comp_dist_km.values())).all():
            
            warnings.warn('No sites are within comparable distance to provided regulatory sites. Bias evaluation will not be performed.')
    
            bias_evaluation = False

    
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
    
    
    #-- Create result tables to add to throughout for loop
    
    # Full bias result table (for potential output)
    full_bias_tbl = pd.DataFrame(columns = ['Standard Pollutant','Site','Regulatory Site','R^2','Slope','Offset','# of Comparison Days','Distance (km)'])
    
    # Full result table to use for seaborn strip plots (excludes duplicate site pair comparisons from precision testing)
    full_result_tbl = pd.DataFrame(columns = ['Standard Pollutant','Site','Metric','Result'])   
    
    # Site-specific median metric table (includes duplicate site pairs from precision testing since this is needed for accurate site-specific median)
    site_median_tbl = pd.DataFrame(columns = ['Standard Pollutant','Site','Metric','Median Result','Site Pair Count'])        
    
    # Average data to daily for bias and precision evaluation (smoothes out data so that we're not focused on intermittent real local pollution differences)
    day_df = _daily_comp(df,config_dict,75)
    
    # Track what pollutants may not meet data criteria to be evaluated for bias or precision (will be used in summary text later)
    bias_p_missing = []
    prec_p_missing = []
    
    
    #-- Loop through pollutant_map in config_dict if bias evaluation or precision evaluation are true
    if bias_evaluation or prec_evaluation:
    
        for p in config_dict['pollutant_map'].keys():
             
            # Define pollutant name as exists in input data and subset day_df
            p_name = config_dict['pollutant_map'][p]
            pdf = day_df[day_df[config_dict['pollutant_col']]==p_name]
            
            # If no data for this pollutant (e.g., user has defined thresholds they don't want to trim to each dataset)
            if not pdf.empty:
                
                # Define what sites are present per pollutant (p_all_sites and p_sites will be the same if no regulatory sites)
                p_all_sites = pdf[config_dict['site_col']].unique()
                p_sites = [ps for ps in p_all_sites if ps in sites]
                
                # If pollutant is not in comp_dist_km, try default and then generic
                if p not in comp_dist_km.keys():
                    
                    try:
                        p_comp_dist = default_comp_dist_km[p]
                        print(f'Warning: No defined comp_dist_km for pollutant {p} ({p_name}) in user-defined settings. Default comp distance used ({p_comp_dist}km).Reminder: you can specify a user-defined comp_dist_km.',end='\n\n')
                    
                    except:
                        p_comp_dist = generic_comp_dist_km
                        print(f'Warning: No defined comp_dist_km for pollutant {p} ({p_name}) in user or default settings. Generic comp distance used ({generic_comp_dist_km}km).Reminder: you can specify a user-defined comp_dist_km.',end='\n\n')
                    
                    # Add this p_comp_dist to the dictionary (will reference later in plots)
                    comp_dist_km[p] = p_comp_dist
                    
                # Otherwise, assign p_comp_dist from set thresholds and check format and value
                else:
                    
                    p_comp_dist = comp_dist_km[p]
        
                    if not isinstance(p_comp_dist, int) | isinstance(p_comp_dist, float): 
                        raise TypeError(f"integer or float expected for comparison distance, got {type(p_comp_dist).__name__} for {p_name}.")
                    
                    elif (p_comp_dist >= 50) & (p_comp_dist < 1000):
                        warnings.warn(f"Comparison distance for {p} is set much higher than default/generic settings and may not result in an accurate performance evaluation.")
                       
                    elif p_comp_dist/1000 >= 1:
                        
                        raise ValueError(f'Comparison distance for {p} is set very high ({p_comp_dist}km), check whether set in correct units (kilometers).')
                        
                # Convert p_comp_dist to float (if not already) and count decimal places to support distance comparisons later
                p_comp_dist = float(p_comp_dist)
                decimal_n = len(str(p_comp_dist).split('.')[1]) # to help with rounding comparison distances
        
                #-- Perform bias_evaluation
                if bias_evaluation:
                    
                    p_regulatory_sites = [r for r in regulatory_sites if r in p_all_sites]
                    
                    if len(p_regulatory_sites)>0 :
                        
                        # Temporary table
                        bias_tbl = pd.DataFrame(columns = ['Standard Pollutant','Site','Regulatory Site','R^2','Slope','Offset','# of Comparison Days','Distance (km)'])
                    
                        # Loop through site in regulatory_sites
                        for r in p_regulatory_sites:
                            
                            r_latlon= [reg_metadata[reg_metadata[config_dict['site_col']]==r][config_dict['lat_col']].values[0],reg_metadata[reg_metadata[config_dict['site_col']]==r][config_dict['lon_col']].values[0]]
                            
                            # Loop through non-regulatory sites
                            for s in p_sites:
                                
                                s_latlon= [sites_metadata[sites_metadata[config_dict['site_col']]==s][config_dict['lat_col']].values[0],sites_metadata[sites_metadata[config_dict['site_col']]==s][config_dict['lon_col']].values[0]]
            
                                distance = round(geodesic(r_latlon, s_latlon).km,decimal_n)
                                
                                # If distance meets comparable distance set for this pollutant, perform linear regression
                                if distance <= p_comp_dist:
                            
                                    b_compare_df = pdf[pdf[config_dict['site_col']].isin([r,s])]
            
                                    # Pivot for easier comparison and remove any rows with NaN - where days are not comparable
                                    b_compare_df = pd.pivot(b_compare_df,index = config_dict['timestamp_col'], columns = config_dict['site_col'], values = config_dict['value_col']).dropna()
                                    
                                    # Check if enough comparison days (at least 25 days - rows)
                                    ns = len(b_compare_df)
                                    
                                    if ns>= 25:
                                    
                                        # Set dependent (y - non-regulatory) and independent variables (x - regulatory)
                                        y=b_compare_df[s]
                                        x=b_compare_df[r]
                
                                        # Calculate r^2, slope, and offset
                                        slope, offset = np.polyfit(x, y, 1)
                                        r2 = round((np.corrcoef(x, y)[0, 1])**2,2)
                                        
                                        # Add results to temporary bias_tbl and round slope and offset
                                        bias_tbl = pd.concat([bias_tbl,pd.DataFrame([{'Standard Pollutant':p,'Site':s,'Regulatory Site':r,'R^2':r2,'Slope':round(slope,2),
                                                        'Offset':round(offset,2),'# of Comparison Days':ns,'Distance (km)':distance}])])
                                        
                                        
                            # No minimum requirement for bias evaluation per pollutant - must have at least one regulatory site, i.e., not an empty dataframe
                                
                            # Add bias results to full_bias_table
                            full_bias_tbl = pd.concat([full_bias_tbl,bias_tbl])
                            
                            # Append full bias_tbl (long format) to full_result_tbl
                            full_result_tbl = (pd.concat([full_result_tbl,
                                                          pd.melt(bias_tbl,id_vars=['Standard Pollutant','Site'],value_vars=['R^2','Slope','Offset'],var_name='Metric',value_name='Result')]))
                            
                            # Calculate median results per site per metric (per pollutant)
                            site_median_tbl = (pd.concat([site_median_tbl,
                                                           pd.melt(bias_tbl,id_vars=['Standard Pollutant','Site'],value_vars=['R^2','Slope','Offset'],var_name='Metric',value_name='Result')
                                                           .groupby(['Standard Pollutant','Site','Metric'])
                                                           .agg(**{'Median Result': ('Result', 'median'),'Site Pair Count': ('Result', 'count')})
                                                           .reset_index()]))
                    else:
                        
                        # Append to list of pollutants that are missing regulatory site data for bias evaluation
                        bias_p_missing.append(p_name)
                        
                
                
                # Perform precision evaluation
                if prec_evaluation:
                    
                    # Temporary table
                    prec_tbl = pd.DataFrame(columns = ['Standard Pollutant','Site','Site B','Metric','Result'])
                    
                    for sa in p_sites:
                        
                        sa_latlon= [sites_metadata[sites_metadata[config_dict['site_col']]==sa][config_dict['lat_col']].values[0],sites_metadata[sites_metadata[config_dict['site_col']]==sa][config_dict['lon_col']].values[0]]
        
                        for sb in p_sites:
        
                            if sb != sa:
                                
                                sb_latlon= [sites_metadata[sites_metadata[config_dict['site_col']]==sb][config_dict['lat_col']].values[0],sites_metadata[sites_metadata[config_dict['site_col']]==sb][config_dict['lon_col']].values[0]]
                                
                                distance = round(geodesic(sa_latlon, sb_latlon).km,decimal_n)
                                
                                if distance <= p_comp_dist:
                                
                                    # Subset dataframe
                                    p_compare_df = pdf[pdf[config_dict['site_col']].isin([sa,sb])]
                                    
                                    # Pivot for easier comparison
                                    p_compare_df = pd.pivot(p_compare_df,index = config_dict['timestamp_col'],columns = config_dict['site_col'], values = config_dict['value_col'])
                                    
                                    # Negative data messes up COD calculation (takes it out of 0-1 range)
                                    p_compare_df[p_compare_df<0] = np.nan
                                    
                                    # Remove any rows with NaN - days are not comparable
                                    p_compare_df = p_compare_df.dropna()
                                    
                                    # Check if enough comparison days (at least 25 days - rows)
                                    ns = len(p_compare_df)
                                    
                                    if ns>= 25:
        
                                        numerator = p_compare_df[sa] - p_compare_df[sb]
                                        denominator = p_compare_df[sa] + p_compare_df[sb]
                                        
                                        cod = round(sqrt(sum((numerator/denominator)**2)/ns),2)
                                    
                                        # Add to full_prec_table
                                        prec_tbl = pd.concat([prec_tbl,pd.DataFrame([{'Standard Pollutant':p,'Site':sa,'Site B':sb,'Metric':'COD','Result':cod}])])
                 
        
                    # Calculate site-specific median for precision metrics and add to site_median_tbl (BEFORE removing site pair duplicates)
                    median_prec_p = (prec_tbl.groupby(['Standard Pollutant','Site','Metric'])
                                     .agg(**{'Median Result': ('Result', 'median'),'Site Pair Count': ('Result', 'count')})
                                     .reset_index())
                
                    #Note: Site-specific results don't need to meet rule of 3 comparisons
                    site_median_tbl = pd.concat([site_median_tbl,median_prec_p])
                    
                    # Append prec_tbl to full_result_tbl (WITHOUT site pair duplicates - to not mess up network site pair count later)
                    prec_tbl['Site Pair'] = prec_tbl[['Site','Site B']].values.tolist()
                    prec_tbl['Site Pair'] = prec_tbl['Site Pair'].apply(lambda x: sorted(x))
                    prec_tbl = prec_tbl.drop_duplicates(subset='Site Pair').drop(['Site Pair'],axis=1)
                    
                    # Check if sufficient # of sites in comparable distance for this pollutant to evaluate precision (# of rows)
                    if len(prec_tbl)>=3:
        
                        full_result_tbl = pd.concat([full_result_tbl,prec_tbl],axis=0)
                        
                    else:
                        # Append to list of pollutants that are missing or don't have enough data for precision evaluation
                        prec_p_missing.append(p_name)
            
            # Ignore pollutants that are in config_dict but not in input data
            else:
                continue
                
    # Proceed with outputs based on full_result_tbl (could be empty if not enough data in comparisons above)
    if not full_result_tbl.empty: 

        # Calculate network median metric result per pollutant and grade against metric criteria with resulting output
        network_median_tbl = (full_result_tbl.groupby(['Standard Pollutant','Metric'])
                              .agg(**{'Median Result': ('Result', 'median'),'Site Pair Count': ('Result', 'count')}).reset_index())
        network_median_tbl = _compare_to_metrics(network_median_tbl,metric_criteria,'Median Result')
        
        #-- Loop through performance evaluation categories and describe results to user in summary text file
        for cat in eval_categories.keys():
            
            nm_df = network_median_tbl[network_median_tbl['Metric'].isin(eval_categories[cat]['metrics'])]
            
            # Begin writing to summary text file
            with open(summary_text_file,"a") as f:
                print(f"{cat} evaluation, answering '{eval_categories[cat]['question']}':",file=f)
                
                # If there is any data for this evaluation category, loop through pollutants and print data use guidance
                if not nm_df.empty:
            
                    for p in nm_df['Standard Pollutant'].unique():
                        
                        p_name = config_dict['pollutant_map'][p]
                        
                        p_dist = comp_dist_km[p]
                        
                        #-- For bias, three possible scenarios: all criteria met, r^2 met but either slope and/or offset not met, r^2 not met
                        if cat == 'Bias':
                            
                            # Pivot dataframe for easier comparison with values equal to boolean from _compare_to_metrics
                            pnm_df = nm_df[nm_df['Standard Pollutant']==p].pivot(index='Standard Pollutant',columns='Metric',values='Meets')
                            
                            # If r^2 met
                            if pnm_df['R^2'].item():
                                
                                # Check if all other criteria met
                                if (pnm_df[['Slope','Offset']]).all().all():
                                    print(f"  {p_name} - Yes, data are accurate (all criteria met ({', '.join(eval_categories[cat]['metrics'])}) with {p_dist}km comparison distance).",file = f)
                                
                                # Slope and/or offset not met
                                else:
                                    print(f'  {p_name} - Maybe, but data correction may be needed (R^2 criteria met, but offset and/or slope criteria outside of ideal range with {p_dist}km comparison distance).',file = f)
                            
                            # R^2 not met
                            else:
                                print(f"  {p_name} - No, data are not accurate (most important criteria, R^2, was not met with {p_dist}km comparison distance).",file = f)

                        if cat == 'Precision':
                            
                            pnm_df = nm_df[nm_df['Standard Pollutant']==p]
                            
                            # Just one metric to check - COD, so don't need to pivot
                            
                            if pnm_df['Meets'].item():
                                
                                if (pnm_df['Median Result'] <= 0.3).item():
                                    print(f"  {p_name} - Yes, data are precise/comparable (COD<=0.3 with {p_dist}km comparison distance).",file = f)
                                
                                # Else, COD is greater than 0.3 but less than the max range (0.45)
                                else:
                                    print(f"  {p_name} - Maybe, but some divergence found on median across sites (0.3  < COD <= 0.45 with {p_dist}km comparison distance). Could be real or a sign of imprecise data.",file = f)
                                
                                
                            # Else, COD is greater than or equal to max range (0.45)
                            else:
                                print(f"  {p_name} - No, data are not precise (COD>={metric_criteria['COD'][1]} with {p_dist}km comparison distance).",file = f)
                    
                    
                    #-- Add text about any pollutants missing data for these evaluations (outside of pollutant for loop)
                    if (cat == 'Bias') and len(bias_p_missing)>0:
                        
                        print(f"  {', '.join(bias_p_missing)} - Could not perform bias evaluation (no regulatory sites with data for pollutant).",file=f)
                        
                    
                    if (cat == 'Precision') and len(prec_p_missing)>0:
                        print(f"  {', '.join(prec_p_missing)} - Could not perform precision evaluation (not enough nearby site comparisons (< 3) with {p_dist}km comparison distance).",file=f)
         
                # If no data in this evaluation category, let user know to look in console output for more info
                else:
                    print(f'  {cat} evaluation could not be performed. See console output for more detail.',file=f)
                
                # Add line break between bias and precision results for text organization     
                print('\n',file=f)
        
        
        #-- Make strip plots summarizing all metric data (one output figure per pollutant)
        for p in full_result_tbl['Standard Pollutant'].unique():
            
            # Define what metrics were calculated per pollutant
            metrics = full_result_tbl.loc[full_result_tbl['Standard Pollutant'] == p,'Metric'].unique()
            
            p_name = config_dict['pollutant_map'][p]
            
            p_dist = comp_dist_km[p]
            
            #-- Set up figure dimensions
            # Number of metrics = number of subplots, determines figure width
            plot_n = len(metrics)
            fig_w = plot_n * 4
            # Margins are in inches
            left_in, right_in = 1.0, 1.7
            
            # Start counting subplot index starting at 1
            iii = 1
            fig = plt.figure(figsize=(fig_w,7))
            plt.rcParams.update({"font.family" : "serif",'font.serif':['Arial']})
    
            # Loop through metrics and add strip plot to figure - use full_result_tbl
            for m in metrics:
                
                tbl = full_result_tbl[(full_result_tbl['Metric'] == m) & (full_result_tbl['Standard Pollutant'] == p)]
                
                # Define min and max of ideal metric range
                m_min, m_max = metric_criteria[m]
                
                #-- Define min and max for y axis depending on metric
                # R^2 and COD have a set possible range of values
                if (m == 'R^2') | (m == 'COD') :
                    min_y = 0
                    max_y = 1
                
                # Slope and offset have an undefined range, base on data/thresholds
                else:
                    min_y = min(m_min,min(tbl['Result'])) - 0.5
                    max_y = max(m_max,max(tbl['Result'])) + 0.5
                
                # Set up subplots based on number of metrics
                ax = plt.subplot(1,plot_n,iii)
                
                # Plot data and configure subplots
                sns.stripplot(tbl,y='Result',size=10,color = 'cornflowerblue',edgecolor='white',linewidth=1)
                plt.ylabel(m,fontsize=16,weight='bold')
                plt.yticks(fontsize=15)
                plt.ylim([min_y,max_y])
                ax.set_xlim([-1,1])
                plt.legend([])
                plt.xticks([])
                
                # Add green shading to indicate ideal range for each metric
                rect = Rectangle((-1,m_min), 2, (m_max-m_min), color='forestgreen',alpha=0.1)
                ax.add_patch(rect)
                
                # Show network median results on plot as a horizontal line
                n_tbl = network_median_tbl[(network_median_tbl['Metric'] == m) & (network_median_tbl['Standard Pollutant']==p)]
                for _,row in n_tbl.iterrows():
                    ax.hlines(row['Median Result'],color='dimgrey',linestyle='dashed',linewidth=2,xmin=-1,xmax=1)
                
                # Add to subplot count
                iii+=1
            
            # Create a custom legend to describe each plot element
            custom_legend = []
            custom_legend.append(patches.Rectangle((0,0),1,1,facecolor='forestgreen',linewidth=0.5,alpha=0.4))
            custom_legend.append(Line2D([0], [0],color='darkgrey', lw=3,linestyle='dashed'))
            custom_legend.append(Line2D([0], [0], marker='o', color='cornflowerblue',linestyle='', markersize=12))
            custom_labels = ['Ideal\nResult Range','Network\nMedian','Individual\nSite Comparison']
            plt.legend(handles=custom_legend,labels=custom_labels,fontsize=12)
            sns.move_legend(ax,"upper left",bbox_to_anchor=(1,1))
            
            # Add comparison distance as text to bottom of plot and description of what metrics inform bias and precision
            sub_text = f'Site-Comparison Distance: {p_dist}km'
            if 'R^2' in metrics:
                sub_text += '\nBias metrics: R^2, Slope, and Offset (comparison to nearby regulatory site(s))'
            if 'COD' in metrics:
                sub_text += '\nPrecision metrics: COD (comparison to nearby site(s))'
            
            # Make more space on the right for the legend, and at the bottom for the text, but remove lefthand white space
            fig.subplots_adjust(bottom=0.15,wspace=0.4,left=(left_in / fig_w),right = (1-right_in/fig_w)) 
            
            # Set up figure title and layout depending on how many subplots there are
            if plot_n>1:
                title = f'Performance Evaluation Results for {p_name}'
                plt.suptitle(title,fontsize=16)
                fig.text(fig.axes[0].get_position().x0,0.03,sub_text,fontsize=12)

            else:
                title = f'Performance Evaluation Results\nfor {p_name}'
                plt.title(title,fontsize=16)
                fig.text(0.01,0.03,sub_text,fontsize=10)

            
            # Save out figure to output file path
            plt.savefig(Path(output_file_path,f'Performance Evaluation Results for {p_name}.png'),dpi=300)
            plt.close(fig)    
            
        # Drop temporary columns from network median table and map user pollutant name to standard pollutant key
        network_median_tbl['Standard Pollutant'] = network_median_tbl['Standard Pollutant'].map(lambda pn: config_dict['pollutant_map'][pn])
        network_median_tbl = network_median_tbl.drop(['Min','Max','Meets'],axis=1).rename(columns = {'Standard Pollutant':'Pollutant'})
        
    # If verbose_reg and bias evaluation true, save out full bias table with regulatory site comparisons
    if verbose_bias_df and bias_evaluation:
        
        full_bias_tbl['Standard Pollutant'] = full_bias_tbl['Standard Pollutant'].map(lambda pn: config_dict['pollutant_map'][pn])
        full_bias_tbl = full_bias_tbl.rename(columns = {'Standard Pollutant':'Pollutant'})
        full_bias_tbl.to_csv(Path(output_file_path,'Verbose Bias Evaluation Results.csv'),index=False)
                        
            
    # Save out site median table and identify any sites where median r^2 or COD are not met
    if not site_median_tbl.empty: 
        
        site_median_tbl['Standard Pollutant'] = site_median_tbl['Standard Pollutant'].map(lambda pn: config_dict['pollutant_map'][pn])
        site_median_tbl = site_median_tbl.rename(columns = {'Standard Pollutant':'Pollutant'})
        site_median_tbl.to_csv(Path(output_file_path,'Median Site-Specific Performance Results.csv'),index=False)
        
        site_median_tbl = _compare_to_metrics(site_median_tbl,metric_criteria,'Median Result')[site_median_tbl['Metric'].isin(['R^2','COD'])]
        site_median_tbl['Does Not Meet'] = ~site_median_tbl['Meets']
        
        # Get list of metric(s) that were flagged for an issue site (combination of r^2 and or COD)
        issue_sites = (site_median_tbl[site_median_tbl['Does Not Meet']]
                       .groupby(['Site','Pollutant'])['Metric'].apply(list)
                       .reset_index())
        
        # List out issue sites in summary text file
        with open(summary_text_file,"a") as f:
            
            if not issue_sites.empty:
                print('The following sites and pollutants do not meet correlation and/or precision criteria (R^2 and COD). Recommend review - there may be issues with siting, instrumentation, data processing, or influence from real local sources.',file=f)
                
                for _,row in issue_sites.iterrows():
                    print(f"  {row['Pollutant']} at {row['Site']} ({', '.join(row['Metric'])})",file=f)
                
                print('Review further detail in Median Site-Specific Performance Results.csv',file=f)
                        
            else:
                print('No issue sites found while reviewing correlation and precision criteria (R^2 and COD).',file=f)
            
            
    #TODO - future version: calculate metrics per month and show how they change over time
    #if monthly_evaluation:
        
        # Calculate COD and r^2 per month and plot to show potential variation     
        
               
    if bias_evaluation or prec_evaluation:  
        
        # Letting user know where output plots and txt file was saved
        print(f'Performance plots, files, and the "Performance Evaluation Summary.txt" have been saved to {output_file_path}.',end='\n\n')
        return network_median_tbl
    
        
    
    # Return empty dataframe since could not perform bias or precision evaluation
    else:
        # Letting user know where output plots and txt file was saved
        print(f'The "Performance Evaluation Summary.txt" have been saved to {output_file_path}.',end='\n\n')
        return pd.DataFrame(columns = ['Pollutant', 'Metric', 'Median Result', 'Site Pair Count'])
        
         
        
    


