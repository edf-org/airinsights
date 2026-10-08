import pandas as pd
import numpy as np
import warnings
from airinsights.helpers import _validate_hourly

def _get_hotspot_type(high_hours, time_bins):
    """Labels the period(s) during the day (morning, midday, evening, night, or all day) when sites have elevated measurements""" 
    labels = [label for label, hours in time_bins.items() if high_hours.isin(hours).any()]
    if set(labels) == set(time_bins):
        return "All day"
    return ", ".join(labels) if labels else "None"

def anomalous_sites(
    input_data: pd.DataFrame, 
    config_dict: dict,
    timeframes:dict = {"30d": pd.Timedelta(days=30), "90d": pd.Timedelta(days=90) , "1y": pd.DateOffset(years = 1)},
    z_thresh : int = 1,  
):
    """Identifies and flags sites in the network with anomolously high pollution levels

    This function identifies sites and hours of day with anomolously high levels of pollution
    using a modified Z-score to compare site-specific hourly means with network-wide hourly means. 
    The function assigns a label based on the hours of day when elevated pollution 
    is detected (Morning, Midday, Evening, Night, All Day, or None). 
    The analysis is always run on all available data ('all time' in the output), and 
    may additionally be run on one or more user-defined timeframes (e.g., the most recent 30 days).

    See examples/anomalous_sites_demo.py on GitHub for a full working example: 
    https://github.com/edf-org/airinsights/blob/main/examples/anomalous_sites_demo.py
    
    Parameters
    ----------
    input_data : pd.DataFrame
        A pandas DataFrame containing AQ data (read in using read_aqdata_file)
    config_dict : dict
        A dictionary containing input parameter names and values. 
    timeframes : dict[str, pd.TimeDelta | pd.DateOffset] | None
        Analysis periods to evaluate in addition to all available data, or 
        `None`, which will only evaluate all available data.
        Dictionary keys are labels that will appear in the output. 
        Dictionary values are either TimeDelta or DateOffset objects, 
        and define lookback periods relative to the most recent 
        datetime in the data. Defaults to 30 day, 90 day, and 1 year timeframes.
    z_thresh : int, default 1
        
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing mean pollutant levels (site_mean) for each site, pollutant, and hour of day for each timeframe, 
        along with the following:
                
            **network_median**: Network-wide median pollutant level for the pollutant, hour of day, and timeframe under consideration
            
            **network_mad**: Network-wide median absolute deviation (MAD) for the pollutant, hour of day, and timeframe under consideration
            
            **z_score_mod**: modified Z-score of the site_mean, calculated using the network-wide median and MAD 
            
            **z_thresh**: Z-score threshold used to determine whether pollutant levels at a site are elevated relative to the network
            
            **elevated**: boolean indicating whether pollutant levels are elevated at the site for the pollutant, hour of day, and timeframe under consideration
            
            **n_hours_elevated**: integer indicating the number of hours when pollutant levels are elevated relative to the network
            
            **times_elevated**: string indicated when elevated levels of a specific pollutant are occurring at the site. 
            Possible values include: "None" indicating levels are not elevated during any time during the day, "All Day" indicating elevated levels 
            during all times of the day, or any combination of "Morning" (6am-11am), "Midday" (11am-5pm), "Evening" (5pm-9pm), or "Night" (9pm-6am)
        
    Notes
    -----
    
    The modified Z-score is calculated as ``abs[(mean_hourly_value - median_value) / 1.4826 * MAD]``, where 1.4826 is a 
    scaling factor used to make MAD comparable to standard deviation. See https://en.wikipedia.org/wiki/Median_absolute_deviation
    """

    df = input_data.copy()
    
    # --- Validate hourly time resolution ---
    df = _validate_hourly(df, config_dict)

    # --- Extract geographic coordinates for all sites to join back later ---
    coords = df[[config_dict['site_col'], config_dict['lat_col'], config_dict['lon_col']]].drop_duplicates()   
     
    # --- Add column for hour of day ---
    df["hour"] = df[config_dict['timestamp_col']].dt.hour
    
    # --- Subset data for multiple timeframes: 30 days, 90 days, 1 year and full dataset ---
    max_time = df[config_dict['timestamp_col']].max()
    
    # Initialize a dictionary for subsets of data, starting with data from the entire period of record
    df_timeframe_dict = {
        "all_time": df
    }
    
    # get unique combinations of site and pollutant 
    site_pol_combos = df[[config_dict['site_col'], config_dict['pollutant_col']]].drop_duplicates()
    
    
    # Check for data completeness for each timeframe, site, and pollutant.
    # If data are sufficiently complete, add to the dictionary. Otherwise, issue a warning.
    for label, timeframe in timeframes.items():
        # calculate the earliest date in the timeframe
        date_lim = max_time - timeframe
        # initialize list to collect combinations of site and pollutant that don't meet the data completeness threshold for the given timeframe
        incomplete_site_pol_combos = []
        # evaluate data completeness for each pollutant in the dataset 
        for pollutant, site in zip(site_pol_combos[config_dict["pollutant_col"]], site_pol_combos[config_dict["site_col"]]):
            # get all data for site-pollutant combination
            pol_site_subset = df[(df[config_dict['pollutant_col']] == pollutant) & (df[config_dict['site_col']] == site)]
            # get all data for site-pollutant combination within timeframe
            tf_site_col_subset = pol_site_subset[pol_site_subset[config_dict['timestamp_col']] >= date_lim]
            # check that beginning of data is before the beginning of the timeframe
            if not pol_site_subset[config_dict['timestamp_col']].min() < date_lim:
                incomplete_site_pol_combos.append({
                    "site": site,
                    "pollutant": pollutant
                })
                #TODO: Consolidate error message e.g., if no data exist at any site for a given pollutant and timeframe, only throw one error message instead of n_sites
                warnings.warn(f"{pollutant} data at site {site} for {label} timeframe for not analyzed because data begin after timeframe start.")
                continue
            # calculate number of valid days (i.e., those with at least 18 hours, or 75% of the day)
            tf_site_col_subset['date'] = tf_site_col_subset[config_dict['timestamp_col']].dt.date
            valid_days = (
                tf_site_col_subset.groupby('date')
                .agg({'hour': 'nunique'})
                .loc[lambda x: x['hour'] >= 18]
            )
            n_valid_days = len(valid_days)
            # calculate total number of days in the timeframe
            n_days = (max_time - date_lim).days
            # check that valid days account for at least 75% of the number of days in the entire timeframe
            #TODO - use generic data completeness check across functions (trends, anomalous sites, etc.)
            if n_valid_days < 0.75*n_days:
                incomplete_site_pol_combos.append({
                                    "site": site,
                                    "pollutant": pollutant
                                })
                warnings.warn(f"{pollutant} data at site {site} for {label} timeframe for not analyzed because data do not meet 75% completeness criteria")
                continue
        # Create a multi-index of site-pollutant combinations  
        remove_idx = pd.MultiIndex.from_frame(pd.DataFrame(incomplete_site_pol_combos))
        tf_subset = df[df[config_dict['timestamp_col']] >= date_lim]
        df_idx = pd.MultiIndex.from_frame(
            tf_subset[[config_dict['site_col'],config_dict['pollutant_col']]]
        )
        # add data that meets completeness criteria to data dictionary
        #TODO: check if data frame is empty and, if so, throw a warning.
        df_timeframe_dict[label] = tf_subset[~df_idx.isin(remove_idx)]
    
    # --- Define hours corresponding to times of day ---
    time_bins = {
        "Morning": list(range(6, 11)),
        "Midday": list(range(11, 17)),
        "Evening": list(range(17, 21)),
        "Night": list(range(21, 24)) + list(range(0, 6))
    }

    results = []

    # --- For each timeframe and pollutant, identify sites with elevated pollution relative to network ---
    for tf_name, df_tf in df_timeframe_dict.items():

        if df_tf.empty:
            continue

        # --- Calculate hourly means for each site and pollutant ---
        mean_val_by_site = (
            df_tf.groupby([config_dict['site_col'],"hour", config_dict['pollutant_col']])[config_dict['value_col']]
            .agg(site_mean="mean")
            .reindex(pd.MultiIndex.from_product( # Include record for all 24 hours at each site
            [df_tf[config_dict['site_col']].unique().tolist(), range(24), df_tf[config_dict["pollutant_col"]].unique().tolist()],
            names=[config_dict['site_col'], "hour", config_dict['pollutant_col']]
            ))
            .reset_index()
        )

        # --- Across all sites, calculate median and median absolute deviation of hourly means for each pollutant---
        stats = (
            mean_val_by_site.groupby(["hour", config_dict['pollutant_col']])["site_mean"]
            .agg(
                network_median="median",
                network_mad=lambda x: np.nanmedian(np.abs(x - np.nanmedian(x)))
            )
            .reset_index()
        )
        mean_val_by_site = mean_val_by_site.merge(stats, on=["hour", config_dict['pollutant_col']], how="left")
        mean_val_by_site["network_mad"] = mean_val_by_site["network_mad"].replace(0, np.nan)

        # --- Calculate modified z-score for each combination of site, hour, and pollutant ---
        mean_val_by_site["z_score_mod"] = (mean_val_by_site["site_mean"] - mean_val_by_site["network_median"]) / (1.4826 * mean_val_by_site["network_mad"])

        # --- Identify sites with elevated pollution and assign hotspot type (e.g., morning, midday, evening, night, or all day) ---
        
        for (_, pollutant), site_data in mean_val_by_site.groupby([config_dict['site_col'], config_dict['pollutant_col']]):
            elevated = site_data["z_score_mod"] > z_thresh
            hotspot = _get_hotspot_type(site_data.loc[elevated, "hour"], time_bins)
            results.append(
                site_data.assign(
                    z_thresh=z_thresh,
                    elevated=elevated,
                    n_hours_elevated=elevated.sum(),
                    times_elevated=hotspot,
                    timeframe=tf_name
                )
            )

    out = pd.concat(results, ignore_index=True)
    out = out.merge(coords,on=config_dict['site_col'],how='left') #bring back coords

    return out