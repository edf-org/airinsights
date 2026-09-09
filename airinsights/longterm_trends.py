import pandas as pd
import pymannkendall as mk
import numpy as np
from airinsights.helpers import _infer_temporal_freq

def _check_stat(stat):
    """Validate stat argument for 'mean', 'median', or a valid percentile string"""
    if stat in ('mean','median'):
        return
    if isinstance(stat,str) and stat.startswith('p') and stat[1:].isdigit() and 0 <= int(stat[1:]) <= 100:
        return
    raise ValueError(f"stat must be 'mean', 'median', or a percentile string like 'p90', got {stat!r}")

def _stat_threshold(x,freq_hours,stat='mean',threshold=0.75):
    if x.empty or len(x) < 2:
        return np.nan
    expected = x.index.days_in_month[0] * 24 / freq_hours # expected hours - adjusts to daily data etc
    actual = x.count() # count non-nulls
    if (actual / expected) < threshold:
        return np.nan
    if stat == 'mean':
        return x.mean()
    if stat == 'median':
        return x.median()
    return x.quantile(int(stat[1:]) / 100)

def _monthly_stat(site_data,config_dict,stat='mean'):
    freq_hours = _infer_temporal_freq(site_data[config_dict['timestamp_col']]).total_seconds() / 3600
    result = (site_data.set_index(config_dict['timestamp_col'])[config_dict['value_col']]
              .resample("MS")
              .apply(lambda x: _stat_threshold(x, freq_hours, stat))
              .reset_index()
              )
    return result

# for seasonal mann-kendall and theil-sen, the important part is to have data for the same month(s) in multiple years
# criteria is at least 75% of months (>= 9 months) have non-NA values for 3 or more years
def _completeness_check(series,min_months=9,min_years_per_month=3):
    valid = series.dropna()
    years_per_month = valid.groupby(valid.index.month).apply(lambda x: x.index.year.nunique())
    complete = (years_per_month >= min_years_per_month).sum() >= min_months
    return complete

def site_trends(input_data,config_dict,stat:str = 'mean',return_data:bool = False):
    """Calculates historical AQ trends by site and pollutant - are pollution levels increasing or decreasing over multiple years?

        This function takes disaggregated AQ measurements, takes monthly averages, and uses statistical tests to determine the
        direction and significance of the trend. Theil-sen and mann-kendall tests are used to determine the magnitude (slope) and
        significance of the trend. To account for seasonality, a seasonal test is applied where each month is compared to the same
        month in past years.

        The function applies data completeness criteria: the same month must have data for at least 3 different years. For annual and network
        results, at least 9 months must have data during at least 3 years.

        Parameters
        ----------
        input_data: pd.DataFrame
            A pandas DataFrame of AQ measurements with site, timestamp, pollutant, and value columns (read in using read_aqdata_file).
        config_dict: dict
            A dictionary containing input parameter names and values.
        stat: str, default 'mean'
            Statistic used to aggregate measurements to a monthly value. One of 'mean', 'median', or a
            percentile string like 'p90' (90th percentile). Useful for tracking how extremes, not just central tendency,
            are changing over time. 
        return_data: bool
            Return the monthly average data? default = False

        Returns
        -------
        network_trend : pd.DataFrame
            A pandas DataFrame containing annual trend results for network monthly mean values
        annual_trend : pd.DataFrame
            A pandas DataFrame containing annual trend results by site and pollutant for the whole duration of the data
        monthly_trend: pd.DataFrame
            A pandas DataFrame containing trend results by site and pollutant, broken out by month of year
        data_out: dict
            Dict of two output datasets for plotting/analysis: "network_monthly" has network mean by month and pollutant,
            "site_monthly" has site mean by month and pollutant

        Notes
        -----
        The network-level trend (network_trend and data_out['network_monthly']) always combines sites by averaging
        each site's monthly stat across sites, regardless of the stat parameter.
        """
    # TODO - reduce redunancy with data audit by moving data capture filters outside

    _check_stat(stat)

    if stat != 'mean':
        print(f"Note: network-level trend uses the mean across sites of each site's monthly {stat}")

    network = []
    annual = []
    monthly = []
    network_data = []
    site_data = []
    
    for pollutant, pollutant_data in input_data.groupby(config_dict['pollutant_col']):

        # calculate monthly means with 75% threshold for valid hours
        df_monthly = (pollutant_data
                      .groupby(config_dict['site_col'])
                      .apply(lambda x: _monthly_stat(x, config_dict, stat))
                      .reset_index(level = config_dict['site_col'])
                      .set_index(config_dict['timestamp_col']))
        df_monthly[config_dict['pollutant_col']] = pollutant
        
        # run seasonal MK/theil-sen on network monthly mean
        monthly_network_mean =  (df_monthly.groupby(df_monthly.index).agg(
            value=(config_dict['value_col'], 'mean'),
            n_sites=(config_dict['site_col'], 'nunique')
        )
        .asfreq('MS')) # ensure monthly data with no gaps
        monthly_network_mean[config_dict['pollutant_col']] = pollutant   
                      
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
            print(f"Skipping network trends for {pollutant}: insufficient seasonal completeness")

        # run seasonal MK/theil-sen
        for site, data in df_monthly.groupby(config_dict['site_col']):
            if _completeness_check(data[config_dict['value_col']]):
                res = mk.seasonal_test(data[config_dict['value_col']], period=12)
                valid = data[config_dict['value_col']].dropna() 
                # add the trend line result to the data
                temp = data.copy()
                position = np.arange(len(temp))
                temp['trend_line'] = res.intercept + res.slope * (position / 12)
                site_data.append(temp)
                # create results dict 
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
            else:
                print(f"Skipping trends for {pollutant} at site {site}: insufficient seasonal completeness")

        # run disagg (on each month with 3 years) MK/theil-sen
        df_monthly['month'] = df_monthly.index.month
        for (site, month), data in df_monthly.groupby([config_dict['site_col'],'month']):
            if data[config_dict['value_col']].count() >= 3: # require >=3 values for a month to calculate trend
                try:
                    res = mk.original_test(data[config_dict['value_col']])
                    monthly.append({
                        config_dict['pollutant_col']: pollutant,
                        config_dict['site_col']: site,
                        'month': month,
                        'trend': res.trend,
                        'slope': res.slope,
                        'intercept': res.intercept,
                        'p_value': res.p}) 
                except Exception as e:
                    print(f"Skipping {site} at {month} for {pollutant}: {e}")
                    continue
        
        network_data.append(monthly_network_mean)
        
    # compile and check results
    network_trend = pd.DataFrame(network)
    annual_trend = pd.DataFrame(annual)
    if annual_trend.empty and network_trend.empty:
        raise ValueError("Data contains no sites or network-level results with sufficient data capture for trend analysis.")
    monthly_trend = pd.DataFrame(monthly)
    
    # add back to original metadata
    metadata = input_data[[config_dict['site_col'], config_dict['lat_col'], config_dict['lon_col']]].drop_duplicates()
    annual_trend = annual_trend.merge(metadata,on = config_dict['site_col'])
    monthly_trend = monthly_trend.merge(metadata,on = config_dict['site_col']) if not monthly_trend.empty else monthly_trend
    
    data_out = {
        'network_monthly': pd.concat(network_data) if network_data else pd.DataFrame(),
        'site_monthly': pd.concat(site_data) if site_data else pd.DataFrame()
    }

    if return_data:
        return network_trend,annual_trend, monthly_trend, data_out
    else:
        return network_trend,annual_trend, monthly_trend