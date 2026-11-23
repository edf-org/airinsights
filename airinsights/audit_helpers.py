
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:20:39 2026


"""
"""
@author: TD Enviro
"""


import pandas as pd
import numpy as np
from airinsights.helpers import _validate_hourly


#-- Default QC threshold checks for audit_data_flagging()

# Range - invalid
default_range_thresholds = {'BC':{'min':-400,'max':500000},'CO':{'min':0,'max':15},'NO2':{'min':-1,'max':500},
                                  'NO':{'min':-1,'max':3000},'O3':{'min':-5,'max':250},'PM2_5':{'min':-5,'max':1000},
                                  'PM10':{'min':-5,'max':10000},'SO2':{'min':-5,'max':300},'VOC':{'min':-1,'max':300}}
# Common error codes - suspect
common_error_codes = [-9999, 865, 895, 995, 985, 999, 999.9, 985, 9999, 99999,1000,2000,3000,4000,5000,6000,7000,8000,9000]

# Rate of change - suspect
default_roc_thresholds = {'NO2':50,'O3':60,'PM2_5':150,'PM10':300}

# Sticking - suspect
default_sticking_hrs_thresholds = {'NO2':{'stck_hr_count':10,'min_val':5},'O3':{'stck_hr_count':12,'min_val':10},
                                         'PM2_5':{'stck_hr_count':6,'min_val':5},'PM10':{'stck_hr_count':6,'min_val':10},
                                         'Generic':{'stck_hr_count':12,'min_val':0}}

# Unrealistic pollutant comparisons, unr_ps: list of [x,y]  where unrealistic pollutant check is for x > y, and Flagging_Reason code needs to match order of [x,y]}
# Note: this does not include unrealistic pollution behvaior for O3 which is defined separately in the audit_data_flagging code documentation
unrealistic_pollutant_comparisons = {'unr_ps':[['PM2_5','PM10'],['NO2','NOx'],['NO','NOx']],'flagging_reason':['UP','U2','UN']}

#-- Standard units for default thresholds
default_threshold_units = {'BC':'ng/m3','CO':'ppm','NO2': 'ppb', 'NO': 'ppb', 'O3': 'ppb', 'PM2_5': 'ug/m3', 'PM10':'ug/m3','SO2': 'ppb','VOC': 'ppm'}

#-- Converting units
def _unit_conversions(x:int | float,unit:str,convert_unit:str,standard_pollutant:str | None = None):
    
    '''
    Converts a unit to another unit (convert_unit). Available options are based on plausaible pollutant concentration units and defaults:
    
    ppm --> ug/m3 (CO ppm default) - standard_pollutant field is needed for these conversions
    
    ppb --> ug/m3 (NO, NO2, SO2 ppb default) - standard_pollutant field is needed for these conversions
    
    ng/m3 --> ug/m3 (BC ng/m3 default)
    
    ug/m3 --> ng/m3 (PM2.5 and PM10 ug/m3 default)
    
    ppb --> ppm (NO, NO2, SO2 ppb default)
    
    ppm --> ppb (CO, VOC ppm default)
    
    Assumptions: gas-phase data will be provided in ppb, ppm, or ug/m3 and PM will be provided in ug/m3 or ng/m3.
    
    '''
    
    #-- Defining what unit pairs are related by a factor of 1000
    mply_1000 = [['ug/m3','ng/m3'],['ppm','ppb']]
    div_1000 = [['ng/m3','ug/m3'],['ppb','ppm']]
    
    #-- Molecular weights (g/mol) for mass/volume concentration conversions (assumes volume of a mole of gas is 24.45 based on 25C and 1atm, also assumes NO2 mw for NOx)
    #TODO - Do we need a mw input field for non standard pollutants (e.g., H2S)?
    mws = {'CO':28.01,'NO2':46.01,'NO':30.01,'NOx':46.01,'O3':48,'SO2':64.06}
    
    #-- Perform unit conversions
    if any([unit,convert_unit] == i for i in mply_1000) :
        return round(x*1000,3)
    
    elif any([unit,convert_unit] == i for i in div_1000) :
        return round(x/1000,3)
    
    elif ([unit,convert_unit] == ['ppb','ug/m3']) | ([unit,convert_unit] == ['ppm','ug/m3']):
        
        # Check if standard_pollutant is in mw dictionary. Don't check this earlier, since could be non-mw related pollutant (e.g., PM2_5) or a non-standard pollutant (e.g., CH4)
        if standard_pollutant not in mws.keys():
            raise ValueError(f'{standard_pollutant} not in available list of molecular weights')
        
        else:
            mw = mws[standard_pollutant]
        
        if unit == 'ppb':
            return round(mw * x / 24.45,3)
        
        elif unit == 'ppm':
            return round((mw * x / 24.45)/1000,3)
    
    elif unit == convert_unit:
        return x
    
    else:
        raise ValueError(f'Conversion between {unit} and {convert_unit} is not available. See function description.')
    
    
#-- Convert default flagging thresholds to user-provided units
def _convert_thresholds(config_dict):
    
    '''
    This function converts all default flagging thresholds from default units to the units provided in the config_dict pollutant_map.
    
    Converted default range, roc, and sticking thresholds are returned (in that order). 
    
    Note: this function only converts threshold values for pollutants that exist in the default threshold dictionaries AND config_dict pollutant_map. 
    All other default thresholds remain unconverted.
    
    '''
        
    #-- Initiate converted threshold dictionaries
    cnv_default_range_thresholds = {}
    cnv_default_roc_thresholds = {}
    cnv_default_sticking_hrs_thresholds = {}
    
    #-- Loop through standard pollutants in config and each default threshold dictionary, building them back up with the conversions
    
    for sp in config_dict['pollutant_map'].keys():
        
        if sp in default_range_thresholds:
            
            # Default and converted min and max values
            rng_min = default_range_thresholds[sp]['min']
            rng_max = default_range_thresholds[sp]['max']
            cnv_min = _unit_conversions(rng_min,default_threshold_units[sp],config_dict['pollutant_map'][sp]['unit'],sp)
            cnv_max = _unit_conversions(rng_max,default_threshold_units[sp],config_dict['pollutant_map'][sp]['unit'],sp)
            
            cnv_default_range_thresholds.update({sp:{'min':cnv_min,'max':cnv_max}})
            
        if sp in default_roc_thresholds:
            
            # Default and converted ROC values
            roc_val = default_roc_thresholds[sp]
            cnv_roc_val = _unit_conversions(roc_val,default_threshold_units[sp],config_dict['pollutant_map'][sp]['unit'],sp)
            
            cnv_default_roc_thresholds.update({sp:cnv_roc_val})

        if sp in default_sticking_hrs_thresholds:
            
            # Default and converted min_val for sticking check (stck_hr_count gets pulled in too)
            sticking_val = default_sticking_hrs_thresholds[sp]['min_val']
            hr_val = default_sticking_hrs_thresholds[sp]['stck_hr_count']
            cnv_sticking_val = _unit_conversions(sticking_val,default_threshold_units[sp],config_dict['pollutant_map'][sp]['unit'],sp)

            cnv_default_sticking_hrs_thresholds.update({sp:{'stck_hr_count':hr_val,'min_val':cnv_sticking_val}})
        
    #-- Combine converted thresholds with rest of default thresholds (if none of the pollutants in the config have default thresholds, the full default thresholds are returned unconverted)
    range_thresholds = default_range_thresholds | cnv_default_range_thresholds
    roc_thresholds = default_roc_thresholds | cnv_default_roc_thresholds
    sticking_hrs_thresholds = default_sticking_hrs_thresholds | cnv_default_sticking_hrs_thresholds
    
    return range_thresholds, roc_thresholds, sticking_hrs_thresholds



#-- Data completeness sub functions to reference throughout other data audit functions and methods


#-- Hourly to complete daily averages
def _daily_comp(input_data:pd.DataFrame,config_dict:dict,comp_threshold:int or float) -> pd.DataFrame:
    
    """
    Average hourly data to complete daily data.
    
    Daily completeness requirement: meets % completeness threshold per day (i.e., # of complete hours per month meet %)
                                                                            
    Assumption: input hourly data represents complete hours.                   
    
    """
    
    hr_count_thresh = round((comp_threshold/100) * 24)
    
    # Check that input data has hourly resolution before proceeding (validate_hourly will throw error)
    df = input_data.copy()
    df = _validate_hourly(df,config_dict)
    
    # Use pandas resample instead of polars group_by_dynamic - resample fills in data gaps with nans which will be needed for accurate monthly heat maps and daily time series (otherwise missing those points on the plots)
    
    # Average hourly data by day and count valid # of hours - by site and pollutant
    day_df = (df.set_index(config_dict['timestamp_col'])
              .groupby([config_dict['site_col'],config_dict['pollutant_col']])
              .resample('D')[config_dict['value_col']]
              .agg(['mean','count'])
              .rename(columns = {'mean':config_dict['value_col']})
              .reset_index())

    day_df[config_dict['value_col']] = np.where(day_df['count']<hr_count_thresh,np.nan,day_df[config_dict['value_col']])

    day_df = day_df.drop('count',axis=1).sort_values(by=[config_dict['site_col'],config_dict['pollutant_col'],config_dict['timestamp_col']])
    
    return day_df



#-- Daily to complete monthly averages
def _month_comp(input_data:pd.DataFrame,config_dict:dict,comp_threshold:int or float) -> pd.DataFrame:
    
    """
    Average daily data to complete monthly data.
    
    Monthly completeness requirement: meets % completeness threshold per month (i.e., # of complete days per month meet %)
    
    """
    
    # Check that input data has daily resolution and throw error otherwise
    df = input_data.copy()
    freq_check = df.set_index(config_dict['timestamp_col']).groupby([config_dict['site_col'],config_dict['pollutant_col']]).resample('D')[config_dict['value_col']].count().max()
    
    if freq_check != 1:
        
        raise ValueError('Input data for _month_comp must have daily resolution.')
        

    # Define the start and end of the complete month range of the dataset to get accurate total expected days per month to compare to for completeness (year is important due to leap years)
    date_min = df[config_dict['timestamp_col']].min() 
    date_max = df[config_dict['timestamp_col']].max() 
    # Create date_range with MS frequency
    months_lst = pd.date_range(date_min.replace(day=1),date_max.replace(day=1), freq='MS').tolist()
    
    # Convert the list of months into a period index to define total expected days in month
    month_ref = pd.DataFrame(months_lst,columns=[config_dict['timestamp_col']])
    month_ref['Days_per_Month'] = (month_ref[config_dict['timestamp_col']].dt.days_in_month)
    
    # Average daily data by month and count valid # of days - by site,pollutant,lat,lon and any other extra identifying columns
    month_df = (df.set_index(config_dict['timestamp_col'])
              .groupby([config_dict['site_col'],config_dict['pollutant_col']])
              .resample('MS')[config_dict['value_col']]
              .agg(['mean','count'])
              .rename(columns = {'mean':config_dict['value_col']})
              .reset_index()
              .merge(month_ref, on = config_dict['timestamp_col'],how='right')) #how=right to ensure that any months missing from the dataset get put back in as nan
    
    
    # NaN any monthly averages that do not meet completeness threshold
    month_df[config_dict['value_col']] = np.where(month_df['count'] < round((comp_threshold/100)*month_df['Days_per_Month']),np.nan,month_df[config_dict['value_col']])

    # Remove temporary columns and sort dataframe
    month_df = month_df.drop(['count','Days_per_Month'],axis=1).sort_values(by=[config_dict['site_col'],config_dict['pollutant_col'],config_dict['timestamp_col']]).reset_index(drop=True)

    
    return month_df
    
    
#-- Monthly to complete 12-month averages
def _annual_comp(input_data:pd.DataFrame,config_dict:dict,comp_threshold:int or float,gap_test:bool=True) -> pd.DataFrame:
    
    """
    Average monthly data to complete annual data.
    
    Annual = consecutive 12-month groups
    
    Annual completeness requirement: meets % completeness threshold AND if gap_test True, must not have more than two consecutive months of incomplete data (proxy for seasonal completeness)

    """
    
    # Check that input data has monthly resolution and throw error otherwise
    df = input_data.copy()
    
    freq_check = df.set_index(config_dict['timestamp_col']).groupby([config_dict['site_col'],config_dict['pollutant_col']]).resample('MS')[config_dict['value_col']].count().max()
    
    if freq_check != 1:
        
        raise ValueError('Input data for _annual_comp must have monthly resolution.')
    
    month_count_thresh = round((comp_threshold/100)*12)    
    
    #-- Define dataset's monthly range and make adjustments to ensure it is uniform across sites,pollutants and contains full 12-month groups
    
    # Define the dataset's month range based on min, max timestamp. Purpose:
        # Sets a uniform date range per site with reindex later (i.e., doesn't maintain any nuance for sites that were installed/desintalled on a different timeline than the min/max)
        # Helps fill in any months missing within the range, ensuring 12-month groups are made of consecutive months
        #^This is especially important for rolling gap_test. Ex. dataset can come in with 9 months of the year and no blanks for the tail end and the gap test would pass (incorrectly)
    full_month_range = pd.date_range(
    start=df[config_dict['timestamp_col']].min(),
    end=df[config_dict['timestamp_col']].max(),
    freq='MS'  
    )
    
    # Check whether df, as is, spans full groups of 12 months (i.e., is divisible by 12 without a remainder)
    grp_check_remainder = len(full_month_range) % 12
    
    # Supplement full_month_range with any missing tail months (making it fully divisible by 12 - i.e., ensures there are full 12-month groups in the range)
    if grp_check_remainder != 0:
        
        full_month_range = pd.date_range(full_month_range[0],periods=len(full_month_range) + (12 - grp_check_remainder),freq='MS')
        
    #-- Reindex df with the full_month_range to ensure that each site,pollutant has a record per month, so that annual completeness checks are accurate
    # We can't just reindex with full_month_range, and need to create a new full multi-index with site,pollutant
    
    # Create unique list of site and pollutant values for multi-index
    index_cols = [config_dict['site_col'],config_dict['pollutant_col']]
    unique_idx = [df[cl].unique() for cl in index_cols]
    
    # Define actual site, pollutant pairs to subset by later (not all sites may measure the same pollutants, but reindex loses this detail)
    sp_pairs = df[index_cols].drop_duplicates()
   
    # Build a multi-index based on unique sites,pollutants, and full_month_range
    full_index = pd.MultiIndex.from_product(unique_idx + [full_month_range],names = index_cols + [config_dict['timestamp_col']])
    
    # Reindex df with new multi-index (fills in any missing months per site,pollutant with nan value)
    df = df.set_index(index_cols+[config_dict['timestamp_col']])[config_dict['value_col']].reindex(full_index).reset_index()
    
    # Trim df back to expected site,pairs (not all sites may measure the same pollutants)
    df = df.merge(sp_pairs, on=index_cols, how='right')
    
    #-- Set up completeness related columns
    
    # Incomp = integer column (1,0) in month_df where 1 = True, incomplete month
    df['Incomp'] = df[config_dict['value_col']].isna().astype(int)
    
    # Roll_3 = Rolling sum of Incomp column on a 3-month basis (completenss rule means no rolling sums of 3 or more incomplete months are allowed per group of 12 months)
    df['Roll_3']= df.groupby(index_cols).rolling(window=3,min_periods=3)['Incomp'].sum().reset_index(drop=True)
    
    # Season_Check = Seasonal check to flag any 3-month rolling sums of completeness that are 3 or greater
    df['Season_Check'] = df['Roll_3']>=3
    
    #-- Compute annual (12-month) completeness
    
    # Average monthly data to 12-month groups, count # of valid months per group and # of True seasonal checks - by site,pollutant
    annual_df = (df.set_index(config_dict['timestamp_col'])
              .groupby([config_dict['site_col'],config_dict['pollutant_col']])
              .resample('12MS')
              .agg(mean = (config_dict['value_col'],'mean'),
                   count = (config_dict['value_col'],'count'),
                   Season_Check = ('Season_Check','sum'))
              .rename(columns = {'mean':config_dict['value_col']})
              .reset_index())
    
    # Define completeness check as years that have less than the month count threshold
    annual_comp_cond = (annual_df['count'] >= month_count_thresh)
    
    # If gap_test is True, add condition that no more than 2 months in a row can be incomplete to the completeness check
    if gap_test:
        
        annual_comp_cond = annual_comp_cond & (annual_df['Season_Check']==0) #if Season_Check>1, then at least one instance of a consecutive 3-month or greater gap
    
    # NaN any annual averages that are incomplete
    annual_df[config_dict['value_col']] = np.where(annual_comp_cond,annual_df[config_dict['value_col']],np.nan)

    # Remove temporary completeness-related columns
    annual_df = annual_df.drop(['count','Season_Check'],axis=1).sort_values(by=[config_dict['site_col'],config_dict['pollutant_col'],config_dict['timestamp_col']]).reset_index(drop=True)
    
    return annual_df