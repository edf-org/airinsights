# Copyright 2026 Environmental Defense Fund, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pymannkendall as mk
import numpy as np
from airinsights.helpers import _infer_temporal_freq
import warnings
import calendar

def _check_stat(stat):
    """Validate stat argument for 'mean', 'median', or a valid percentile string"""
    
    # validate dtype
    if not isinstance(stat,str):
        raise TypeError(f"stat must be a string, got {type(stat)}")

    # is it a percentile string?
    is_percentile = stat.startswith('p') and stat[1:].isdigit() and 0 <= int(stat[1:]) <= 100

    # raise error if stat is not any of the options
    if stat not in ('mean','median') and not is_percentile:
        raise ValueError(f"stat must be 'mean', 'median', or a percentile string like 'p90', got {stat!r}")

def _stat_threshold(x,
                    freq_hours,
                    stat='mean',
                    threshold=0.75
                    ):
    """For a single month of grouped data, calculate specified statistic with data capture threshold"""
    
    result = np.nan # if no data or below capture threshold, return nan

    # if data, calculate statistic
    if not x.empty:
        expected = x.index.days_in_month[0] * 24 / freq_hours # expected hours - adjusts to daily data etc
        actual = x.count() # count non-nulls
        if (actual / expected) >= threshold:
            if stat == 'mean':
                result = x.mean()
            elif stat == 'median':
                result = x.median()
            else:
                result = x.quantile(int(stat[1:]) / 100)
    return result

def _monthly_stat(site_data,
                  config_dict,
                  stat='mean'
                  ):
    """Resample site timeseries to monthly, using data capture threshold and statistic"""
    freq_hours = _infer_temporal_freq(site_data[config_dict['timestamp_col']]).total_seconds() / 3600
    result = (site_data.set_index(config_dict['timestamp_col'])[config_dict['value_col']]
              .resample("MS")
              .apply(lambda x: _stat_threshold(x, freq_hours, stat))
              .rename('value')
              .reset_index()
              )
    return result

# for seasonal mann-kendall and theil-sen, the important part is to have data for the same month(s) in multiple years
# criteria is at least 75% of months (>= 9 months) have non-NA values for 3 or more years
def _completeness_check(series,
                        min_months=9,
                        min_years_per_month=3
                        ):
    """Validate monthly time series to ensure sufficient data capture for trends method"""
    valid = series.dropna()
    years_per_month = valid.groupby(valid.index.month).apply(lambda x: x.index.year.nunique())
    complete = (years_per_month >= min_years_per_month).sum() >= min_months
    return complete

def annual_trends(input_data: pd.DataFrame,
                config_dict: dict,
                stat:str = 'mean'
                ):
    """Calculates historical AQ trends by site and pollutant - are pollution levels increasing or decreasing over multiple years?

    This function takes disaggregated AQ measurements, takes monthly averages, and uses statistical tests to determine the
    direction and significance of the trend. Theil-Sen and Mann-Kendall tests are used to determine the magnitude (slope) and
    significance of the trend. To account for seasonality, a seasonal test is applied where each month is compared to the same
    month in past years.

    The function applies data completeness criteria: the same month must have data for at least 3 different years. For annual and network
    results, at least 9 months must have data during at least 3 years.

    See examples/annual_trends.ipynb on GitHub for a full working example: 
    https://github.com/edf-org/airinsights/blob/main/examples/annual_trends.ipynb

    Parameters
    ----------
    input_data: pd.DataFrame
        A pandas DataFrame containing AQ data that was read using one of the helpers.read_aqdata_[x] routines.
    config_dict: dict
        A dictionary containing input_data parameters that was read using one of the helpers.read_aqdata_[x] routines.
    stat: str, default 'mean'
        Statistic used to aggregate measurements to a monthly value. One of 'mean', 'median', or a
        percentile string like 'p90' (90th percentile). Useful for tracking how extremes, not just central tendency,
        are changing over time. 

    Returns
    ---------
    network_trend : pd.DataFrame
        A pandas DataFrame with one row per pollutant containing annual trend results for 
        network monthly values, with the following columns:

            **trend**: direction of the trend: "increasing", "decreasing", or "no trend" (based on p_value at the 0.05 significance level)

            **slope**: Theil-Sen estimate of the rate of change in pollutant level per year

            **intercept**: Theil-Sen estimate of the pollutant level at the start of the record (start_month)

            **p_value**: p-value of the seasonal Mann-Kendall test

            **n_sites_min**, **n_sites_median**, **n_sites_max**: minimum, median, and maximum number of sites contributing to the network value across months

            **n_years**: number of years with data

            **n_months**: number of months with data

            **start_month**, **end_month**: first and last month with data

    annual_trend : pd.DataFrame
        A pandas DataFrame with one row per site and pollutant containing annual trend results.
        Has the same columns as network_trend (except for site counts), plus site name and coordinates.

    monthly_trend: pd.DataFrame
        A pandas DataFrame with one row per site, pollutant, and month of year. Shows annual trend
        results for individual months across multiple years (e.g. how have PM2.5 levels in January changed
        over the past five years?). Contains the same core columns as annual_trend, plus a month column.

    data_out: dict
        Dict of two output datasets for plotting/analysis: 

            **network_monthly**: pd.DataFrame
                monthly time series of network value by pollutant, with the columns:

                    **month**: datetime truncated to the first day of the month

                    **value**: mean across sites of the monthly stat

                    **n_sites**: number of sites contributing to stat

                    **pollutant**

                    **trend_line**: fitted trend

            **site_monthly**: pd.DataFrame
                monthly time series of value by site and pollutant, with the same core columns as network_monthly plus a site name column

    Notes
    -------
    The network-level trend (network_trend and data_out['network_monthly']) always combines sites by averaging
    each site's monthly stat across sites, regardless of the stat parameter (which defines the stat for monthly aggregation at each site).
    """
    # TODO - reduce redundancy with data audit by moving data capture filters outside

    # validate the user-input statistic for monthly resampling
    _check_stat(stat)

    if stat != 'mean':
        warnings.warn(f"Note: network-level trend uses the mean across sites of each site's monthly {stat}")

    # initiate outputs
    network = []
    annual = []
    monthly = []
    network_data = []
    site_data = []

    # loop through pollutants to calculate trends
    for pollutant, pollutant_data in input_data.groupby(config_dict['pollutant_col']):

        # calculate monthly means with 75% threshold for valid hours
        df_monthly = (pollutant_data
                      .groupby(config_dict['site_col'])
                      .apply(lambda x: _monthly_stat(x, config_dict, stat))
                      .reset_index(level = config_dict['site_col'])
                      .set_index(config_dict['timestamp_col']))
        df_monthly[config_dict['pollutant_col']] = pollutant
        
        # run seasonal MK/theil-sen on network monthly mean
        monthly_network_mean = (df_monthly.groupby(df_monthly.index).agg(  
            value = ('value', 'mean'),  
            n_sites = ('value', 'count')  
        )
        .asfreq('MS')) # ensure monthly data with no gaps
        monthly_network_mean[config_dict['pollutant_col']] = pollutant   

        # if there is sufficient data, calculate network trends              
        if _completeness_check(monthly_network_mean['value']):
            res = mk.seasonal_test(monthly_network_mean['value'], period=12)
            valid = monthly_network_mean['value'].dropna() 
            # add the trend line result to the data
            position = np.arange(len(monthly_network_mean))
            monthly_network_mean['trend_line'] = res.intercept + res.slope * (position / 12)
            network.append({
                config_dict['pollutant_col']: pollutant,
                'trend': res.trend,
                'slope': res.slope,
                'intercept': res.intercept,
                'p_value': res.p,
                'n_sites_min': monthly_network_mean['n_sites'].min(),
                'n_sites_median': monthly_network_mean['n_sites'].median(),
                'n_sites_max': monthly_network_mean['n_sites'].max(),
                'n_years': valid.index.year.nunique(),
                'n_months': valid.count(),
                'start_month' : valid.index.min(),
                'end_month' : valid.index.max()
            })
        else:
            warnings.warn(f"Skipping network trends for {pollutant}: insufficient seasonal completeness")
        network_data.append(monthly_network_mean) # append network data to return

        # if there is sufficient data, calculate site trends   
        for site, data in df_monthly.groupby(config_dict['site_col']):
            if _completeness_check(data['value']):
                res = mk.seasonal_test(data['value'], period=12)
                valid = data['value'].dropna() 
                # add the trend line result to the data
                position = np.arange(len(data))
                trend_line = res.intercept + res.slope * (position / 12)
                annual.append({
                    config_dict['pollutant_col']: pollutant,
                    config_dict['site_col']: site,
                    'trend': res.trend,
                    'slope': res.slope,
                    'intercept': res.intercept,
                    'p_value': res.p,
                    'n_years': valid.index.year.nunique(),
                    'n_months': valid.count(),
                    'start_month' : valid.index.min(),
                    'end_month' : valid.index.max()
                })
                site_data.append(data.assign(trend_line=trend_line)) # append site data to return
            else:
                warnings.warn(f"Skipping trends for {pollutant} at site {site}: insufficient seasonal completeness")  

        # if there is sufficient data, calculate site trends disaggregated by month of year  
        df_monthly['month'] = df_monthly.index.month
        for (site, month), data in df_monthly.groupby([config_dict['site_col'],'month']):
            if data['value'].count() >= 3: # require >=3 values for a month to calculate trend
                res = mk.original_test(data['value'])
                monthly.append({
                    config_dict['pollutant_col']: pollutant,
                    config_dict['site_col']: site,
                    'month': month,
                    'trend': res.trend,
                    'slope': res.slope,
                    'intercept': res.intercept,
                    'p_value': res.p})
            else:
                warnings.warn(f"Skipping {calendar.month_abbr[month]} trends for {pollutant} at site {site}: fewer than 3 years of data")
        
    # compile and check site results
    network_trend = pd.DataFrame(network)
    annual_trend = pd.DataFrame(annual)
    if annual_trend.empty and network_trend.empty:
        raise ValueError("Data contains no sites or network-level results with sufficient data capture for trend analysis.")
    monthly_trend = pd.DataFrame(monthly)
    
    # add back original metadata
    metadata = input_data[[config_dict['site_col'], config_dict['lat_col'], config_dict['lon_col']]].drop_duplicates()
    annual_trend = annual_trend.merge(metadata,on = config_dict['site_col'])
    monthly_trend = monthly_trend.merge(metadata,on = config_dict['site_col']) if not monthly_trend.empty else monthly_trend

    # assemble output data
    data_out = {
        'network_monthly': pd.concat(network_data).reset_index(names='month') if network_data else pd.DataFrame(),
        'site_monthly': pd.concat(site_data).reset_index(names='month') if site_data else pd.DataFrame()
    }

    return network_trend, annual_trend, monthly_trend, data_out