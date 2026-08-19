import pandas as pd
import numpy as np

def get_hotspot_type(high_hours, time_bins):
    labels = [label for label, hours in time_bins.items() if high_hours.isin(hours).any()]
    if set(labels) == set(time_bins):
        return "All day"
    return ", ".join(labels) if labels else "None"

def anomalous_sites(
    input_data: pd.DataFrame, 
    config_dict: dict,
    z_thresh : int = 1,  
):
    """Identifies and flags sites in the network with anomolously high pollution levels

    This function calculates mean hourly pollutant levels for each site in the network for multiple timeframes: 
    30 days, 90 days, 1 year, and the entire period of record. For each of these timeframes, it identifies sites and hours of day with 
    anomolously high levels of pollution using a modified Z-score to compare site-specific hourly means with network-wide hourly means. 
    For each site, the function assigns a label based on the hours of day when elevated pollution is detected at each site 
    (Morning, Midday, Evening, Night, All Day, or None).

    See examples/anomalous_sites_demo.py on GitHub for a full working example: 
    https://github.com/edf-org/airinsights/blob/main/examples/anomalous_sites_demo.py
    
    Parameters
    ----------
    input_data : pd.DataFrame
        A pandas DataFrame containing AQ data
    config_dict : dict
        A dictionary containing input parameter names and values. See 'Other Parameters' for a list
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
            
                    

    Other Parameters
    ----------------
    site_col : str or int
        Name of the column containing unique identifiers for the air sensors
    lat_col: str 
        Name of column containing site latitude
    lon_col: str
        Name of column containing site longitude
    timestamp_col : str
        Name of the column containing dates and times for each measurement
    value_col : str
        Name of the column containing measurement values
    pollutant_col : str
        Name of the column containing the pollutant name
        
    Notes
    -----
    
    The modified Z-score is calculated as ``abs[(mean_hourly_value - median_value) / 1.4826 * MAD]``, where 1.4826 is a 
    scaling factor used to make MAD comparable to standard deviation. See https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    
    # --- Deep copy input data to avoid unintentionally alterring original ---
    df = input_data.copy()

    # --- Extract geographic coordinates for all sites to join back later ---
    coords = df[[config_dict['site_col'], config_dict['lat_col'], config_dict['lon_col']]].drop_duplicates()   
     
    # --- Add column for hour of day ---
    df["hour"] = df[config_dict['timestamp_col']].dt.hour

    # --- TODO: Check for data completeness within each subset ---
    # check that data doesn't start or end within range (+- 1-2 days)
    # valid day = 75% of hours
    # valid time range = 75% of days in time range
    # Repeated gaps during same part of day : leave for future version
    
    # --- Subset data for multiple timeframes: 30 days, 90 days, 1 year and full dataset ---
    max_time = df[config_dict['timestamp_col']].max()
    timeframes = {
        "30d": df[df[config_dict['timestamp_col']] >= max_time - pd.Timedelta(days=30)],
        "90d": df[df[config_dict['timestamp_col']] >= max_time - pd.Timedelta(days=90)],
        "1y": df[df[config_dict['timestamp_col']] >= max_time - pd.DateOffset(years=1)],
        "all_time": df
    }
    
    # --- Define hours corresponding to times of day ---
    time_bins = {
        "Morning": list(range(6, 11)),
        "Midday": list(range(11, 17)),
        "Evening": list(range(17, 21)),
        "Night": list(range(21, 24)) + list(range(0, 6))
    }

    results = []

    #TODO: group by pollutant
    # --- For each timeframe and pollutant, identify sites with elevated pollution relative to network ---
    for tf_name, df_tf in timeframes.items():

        if df_tf.empty:
            continue

        # --- Calculate hourly means for each site and pollutant ---
        mean_val_by_site = (
            df_tf.groupby([config_dict['site_col'],"hour"])[config_dict['value_col']]
            .agg(site_mean="mean")
            .reindex(pd.MultiIndex.from_product( # Include record for all 24 hours at each site
            [df_tf[config_dict['site_col']].unique().tolist(), range(24)],
            names=[config_dict['site_col'], "hour"]
            ))
            .reset_index()
        )

        # --- Across all sites, calculate median and median absolute deviation of hourly means for each pollutant---
        stats = (
            mean_val_by_site.groupby("hour")["site_mean"]
            .agg(
                network_median="median",
                network_mad=lambda x: np.nanmedian(np.abs(x - np.nanmedian(x)))
            )
            .reset_index()
        )
        mean_val_by_site = mean_val_by_site.merge(stats, on="hour", how="left")
        mean_val_by_site["network_mad"] = mean_val_by_site["network_mad"].replace(0, np.nan)

        # --- Calculate modified z-score for each combination of site, hour, and pollutant ---
        mean_val_by_site["z_score_mod"] = (mean_val_by_site["site_mean"] - mean_val_by_site["network_median"]) / (1.4826 * mean_val_by_site["network_mad"])

        # --- Identify sites with elevated pollution and assign hotspot type (e.g., morning, midday, evening, night, or all day) ---
        
        for site, site_data in mean_val_by_site.groupby(config_dict['site_col']):
            elevated = site_data["z_score_mod"] > z_thresh
            hotspot = get_hotspot_type(site_data.loc[elevated, "hour"], time_bins)
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