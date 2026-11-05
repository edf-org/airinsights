
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:20:39 2026


"""
"""
@author: TD Enviro
"""

# Data completeness sub functions to reference throughout other data audit functions and methods

import polars as pl
import pandas as pd
from datetime import timedelta
import numpy as np



# Hourly to daily
def _daily_comp(df:pd.DataFrame,config_dict:dict,comp_threshold:int) -> pd.DataFrame:
    """Average 1hr data to complete daily data."""
    
    hr_count_thresh = round((comp_threshold/100) * 24)
    
    # day_df = (pl.from_pandas(df).group_by_dynamic(index_column=config_dict['timestamp_col'],every='1d',group_by=[config_dict['site_col'],config_dict['pollutant_col']])
    #     .agg(
    #         Value = pl.col(config_dict['value_col']).mean(),
    #         Count= pl.col(config_dict['value_col']).count()
    #     )).to_pandas().rename(columns = {'Value':config_dict['value_col']})
    
    # Use pandas resample instead of polars group_by_dynamic - resample fills in data gaps with nans which will be needed for accurate monthly heat maps and daily time series (otherwise missing those points on the plots)
    
    # Average hourly data by month and count valid # of hours - by site,pollutant,lat,lon and any other extra identifying columns
    #TODO - make sure that including lat,lon in group_cols doesn't break averaging (i.e.,lat lon isn't changing per site) - address this in read_aqdata_file
    group_cols = [c for c in df.columns if c not in [config_dict['value_col'], config_dict['timestamp_col']]]
    day_df = (df.set_index(config_dict['timestamp_col'])
              .groupby(group_cols)
              .resample('D')[config_dict['value_col']]
              .mean()
              .reset_index())
    
    count = (df.set_index(config_dict['timestamp_col'])
             .groupby([config_dict['site_col'],config_dict['pollutant_col']])
             .resample('D')[config_dict['value_col']]
             .count()
             .reset_index()
             .rename(columns = {config_dict['value_col']:'Count'}))
    
    day_df = pd.merge(day_df,count,on=[config_dict['timestamp_col'],config_dict['site_col'],config_dict['pollutant_col']])


    day_df[config_dict['value_col']] = np.where(day_df['Count']<hr_count_thresh,np.nan,day_df[config_dict['value_col']])

    day_df = day_df.drop('Count',axis=1).sort_values(by=[config_dict['site_col'],config_dict['pollutant_col'],config_dict['timestamp_col']])
    
    return day_df



# Complete monthly data
def _month_comp(df:pd.DataFrame,config_dict:dict,comp_threshold:int) -> pd.DataFrame:
    """Average daily data to complete monthly data."""
    

    # Define the start and end of the complete month range of the dataset to get accurate total expected days per month to compare to for completeness (year is important due to leap years)
    # Need to define the date range from beginning of first data month to last day of last data month to assign completeness fully - can find last day of the last month using replace() and defining next following month
    date_min = df[config_dict['timestamp_col']].min() # Assumes df timestamp_col is already a datetime object
    date_max = df[config_dict['timestamp_col']].max() 
    nxt_month = date_max.replace(day=28) + timedelta(days=4)
    month_last_day = nxt_month - timedelta(days=nxt_month.day)
    # Create date_range with ME frequency
    months_lst = pd.date_range(date_min.replace(day=1),month_last_day, freq='MS').tolist()
    
    # Convert the list of months into a period index to define total expected days in month
    month_ref = pd.DataFrame(months_lst,columns=[config_dict['timestamp_col']])
    periods = pd.PeriodIndex(month_ref[config_dict['timestamp_col']], freq = 'M')
    month_ref['Days_per_Month'] = periods.days_in_month
    
    # Average daily data by month and count valid # of days - by site,pollutant,lat,lon and any other extra identifying columns
    #TODO - make sure that including lat,lon in group_cols doesn't break averaging (i.e.,lat lon isn't changing per site) - address this in read_aqdata_file
    group_cols = [c for c in df.columns if c not in [config_dict['value_col'], config_dict['timestamp_col']]]
    month_df = ((pl.from_pandas(df).group_by_dynamic(index_column=config_dict['timestamp_col'],every='1mo',group_by=group_cols)
                 .agg(
                     Value = pl.col(config_dict['value_col']).mean(),
                     Count= pl.col(config_dict['value_col']).count()
                     )).to_pandas()
                .rename(columns = {'Value':config_dict['value_col']})
                .merge(month_ref, on = config_dict['timestamp_col'],how='right')) #how=right to ensure that any months missing from the dataset get put back in as nan
    
    
    # NaN any monthly averages that do not meet completeness threshold
    month_df[config_dict['value_col']] = np.where(month_df['Count'] < round((comp_threshold/100)*month_df['Days_per_Month']),np.nan,month_df[config_dict['value_col']])

    # Remove temporary columns and sort dataframe
    month_df = month_df.drop(['Count','Days_per_Month'],axis=1).sort_values(by=[config_dict['site_col'],config_dict['pollutant_col'],config_dict['timestamp_col']]).reset_index(drop=True)

    
    return month_df
    
    
# Annual comp based on monthly comp
def _annual_comp(input_data:pd.DataFrame,config_dict:dict,comp_threshold:int,gap_test:bool=True) -> pd.DataFrame:
    """Average monthly data to complete annual data - per calendar year."""
    
    # Annual completeness requirement: meets completeness threshold, and no more than two consecutive months of incomplete data (proxy for seasonal completeness) - this latter requirement may be turned off by user
    month_count_thresh = round((comp_threshold/100)*12)
    
    df = input_data.copy()
    
    # Make sure that all months of year are in month_df - and make Value nan since incomplete/not present (note: end may be in future)
    #^^This is important especially for rolling gap_test. Ex. dataset can come in with 9 months of the year and no blanks for the tail end and the gap test would pass (incorrectly)

    # Build the full date range (e.g. full calendar years)
    full_month_range = pd.date_range(
    start=df[config_dict['timestamp_col']].min().replace(month=1, day=1),
    end=df[config_dict['timestamp_col']].max().replace(month=12, day=1),
    freq='MS'  # month start; use 'M' for month end if your dates are end-of-month
    )

    # Define the columns to index the dataset by and create unique list of their values (for MultiIndex)
    index_cols = [config_dict['site_col'],config_dict['pollutant_col']]#[c for c in df.columns if c not in [config_dict['value_col'],config_dict['timestamp_col']]]
    unique_idx = [df[cl].unique() for cl in index_cols]
    # Define site-pollutant pairs that should exist in the data (reindexing loses this)
    sp_pairs = df[index_cols].drop_duplicates()
    # Define the other columns in the dataset (that are not site, pollutant, timestamp, or value) - we'll add these back in later
    other_cols = [c for c in df.columns if c not in index_cols + [config_dict['timestamp_col'],config_dict['value_col']]]
    # Build a lookup for the extra columns in the dataset
    other_col_lookup = df[index_cols + other_cols].drop_duplicates(subset=index_cols)
    # Build a multiindex based on 
    full_index = pd.MultiIndex.from_product(unique_idx + [full_month_range],names = index_cols + [config_dict['timestamp_col']])
    # Reindex df with new full index which includes all months in each calendar year
    df = df.set_index(index_cols+[config_dict['timestamp_col']])[config_dict['value_col']].reindex(full_index).reset_index()
    # Merge the other cols back in based on site,pollutant
    df = df.merge(other_col_lookup, on=index_cols, how='left')
    # Trim df back to expected site,pairs -- allowing for some sites to not measure all pollutants
    df = df.merge(sp_pairs, on=index_cols, how='right')
    
    # Create completeness related columns
    
    # Create integer column (1,0) in month_df where 1 = incomplete month
    df['Incomp'] = df[config_dict['value_col']].isna().astype(int)
    # Take rolling sum of Comp column on a 3 month basis (completenss rule means no rolling sums of 3 or more incomplete months are allowed per year)
    df['Roll_3']= df.groupby([config_dict['site_col'],config_dict['pollutant_col']]).rolling(window=3,min_periods=3)['Incomp'].sum().reset_index(drop=True)
    # Create Seasonal Check to flag any 3 month rolling completeness sums that are 3 or greater
    df['Season_Check'] = df['Roll_3']>=3
    
    # Average monthly data to year and count # of valid months per year - by site,pollutant,lat,lon and any other extra identifying columns
    #TODO - make sure that including lat,lon in group_cols doesn't break averaging (i.e., lat lon isn't changing per site)
    group_cols = [c for c in df.columns if c not in [config_dict['value_col'], config_dict['timestamp_col'],'Season_Check','Roll_3','Incomp']]
    annual_df = (pl.from_pandas(df).group_by_dynamic(index_column=config_dict['timestamp_col'],every='1y',group_by=group_cols)
        .agg(
            Value = pl.col(config_dict['value_col']).mean(),
            Count = pl.col(config_dict['value_col']).count(),
            Season_Check = pl.col('Season_Check').sum(), 
        )).to_pandas().rename(columns = {'Value':config_dict['value_col']}).sort_values(by=[config_dict['site_col'],config_dict['pollutant_col'],config_dict['timestamp_col']]).reset_index(drop=True)
    
    # Define completeness check as years that have less than the month count threshold
    annual_comp_cond = (annual_df['Count'] >= month_count_thresh)
    
    #If gap_test is True, add condition that no more than 2 months in a row can be incomple to completeness check
    if gap_test:
        
        annual_comp_cond = annual_comp_cond & (annual_df['Season_Check']==0) #if Season_Check>1, then at least one instance of 3 month or greater gap
    
    # NaN any annual averages that are incomplete
    annual_df[config_dict['value_col']] = np.where(annual_comp_cond,annual_df[config_dict['value_col']],np.nan)

    # Remove temporary columns
    annual_df = annual_df.drop(['Count','Season_Check'],axis=1).sort_values(by=[config_dict['site_col'],config_dict['pollutant_col'],config_dict['timestamp_col']]).reset_index(drop=True)
    
    return annual_df