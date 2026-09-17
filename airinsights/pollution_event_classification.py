# ===============================================================
# Classify pollution events as 'Regional' or 'Local' and group local events in space and time using DBSCAN clustering
# ===============================================================

# --- Import packages and functions ---
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import geopandas as gpd
import warnings
from airinsights.pollution_event_detection import pollution_event

def classify_pollution_events(input_data: pd.DataFrame,
                              config_dict: dict,
                              window_size : int = 60,
                              z_thresh : int = 2,
                              local_distance_km : int = 5
                              ):
    """Flags pollution events from AQ measurements and groups into 'Regional' and 'Local' events

    This function runs the pollution_event function internally to flag anomalous measurements, then groups them
    into more interpretable 'Regional' and 'Local' events. 'Regional' (i.e. the extent of the network) events are identified
    when the network median z-score (global Z) exceeds the defined Z threshold (default is 2). 'Local' events are identified when
    the difference between the local (site) Z and the global Z exceeds the defined Z threshold.

    For 'Local' events, 3-dimensional DBSCAN is tuned with a defined spatial eps (default: 5 km) and a temporal eps of 4 hours.
    This groups events close enough in space and time that they could be associated with the same local emissions event.

    This function relies on network statistics to differentiate local and regional events. We apply the following logic:
    * If the input dataset contains data for 3 or more monitors:
        * The function runs as described above
        * If any individual hours have less than 3 sites with data, they are excluded from classification
    * If the input dataset contains data for less than 3 monitors:
        * The function will return a warning and cluster pollution events
        using only site-specific Z scores (no network subtraction for regional events).

    Parameters
    ----------
    input_data: pd.DataFrame
        A pandas DataFrame containing AQ data. Passed directly to the pollution_event function to flag anomalous measurements.
    config_dict: dict
        A dictionary containing input parameter names and values.
    window_size : int, default 60
        Number of days in the rolling window used for pollution_event calculations. Passed through to the pollution_event function.
    z_thresh : int
        Threshold for modified z score to include measurements in event classification. Default value of 2 ("Unusually high").
    local_distance_km : int
        Eps (in km) for local DBSCAN run; the greatest distance at which monitors are grouped as neighbors. Default value of 5 km.
    Returns
    -------
    summary : pd.DataFrame
        A pandas DataFrame with one row per classified pollution event, with the following columns:

            **event_ID**: unique identifier for the event, prefixed by scale and pollutant (e.g. "REG_PM2.5_0", "LOC_PM2.5_3")

            **start_time**, **end_time**: start and end times of the pollution event

            **event_scale**: "Regional" or "Local"

            **sites**: "Network-wide" for Regional events, or a comma-separated list of sites for Local events

            **peak_value**: maximum measured value observed during the event

            **typical_value_at_peak**: the network's (Regional) or site's (Local) typical value for the peak's hour of day, for comparison against peak_value

    df: pd.DataFrame
        A pandas DataFrame containing the site, timestamp, and pollutant columns from the input data with the following columns appended:

            **median_at_hour**: measurement median for hour of day (from pollution_event function)
            
            **z_score_mod**: modified Z-score of the sensor measurement (from pollution_event function)
            
            **local_z**: the site's z-score, adjusted for the network median z-score (if at least 3 sites have data)
            
            **network_median_value**: network-wide median measured value at that timestamp

            **network_median_z**: network-wide median z-score at that timestamp

            **network_typical_value**: network-wide median of each site's typical value for that hour of day

            **event_ID**: list of pollution event_ID's associated with that site/timestamp.
    """

    # --- Run pollution event detection to flag anomalous measurements ---
    # pollution_event does not return value/lat/lon columns, so join them back from the raw input
    flagged = pollution_event(input_data, config_dict, verbose=True, window_size=window_size)
    join_cols = [config_dict['site_col'], config_dict['timestamp_col'], config_dict['pollutant_col']]
    input_data = flagged.merge(
        input_data[join_cols + [config_dict['value_col'], config_dict['lat_col'], config_dict['lon_col']]],
        on=join_cols,
        how='left'
    )

    summary_list_all = []
    df_list_all = []
    
    # --- Loop over each pollutant ---

    for pollutant in input_data[config_dict['pollutant_col']].unique():

        # --- Read input data as df ---
        df = input_data[input_data[config_dict['pollutant_col']] == pollutant].copy()

        # --- calculate hours from start as time dimension ---
        df['hours'] = (df[config_dict['timestamp_col']] - df[config_dict['timestamp_col']].min()).dt.total_seconds() / 3600 

        # --- Quality checks ---
        df['sites_per_hour'] = df.groupby(config_dict['timestamp_col'])['z_score_mod'].transform('count') # count non-NA site z scores by hour
        median_sites = df['sites_per_hour'].median()
        
        # --- Create list to append with stats for different event types ---
        events_list = []
        summary_list = []
        
        if median_sites >= 3:
            print(f"Median of {median_sites} sites per timestamp with {pollutant} data, continuing with regional vs. local classification")
            regional = True
            
            if any(df['sites_per_hour'] < 3):
                print("Some timestamps have less than 3 site measurements and will be skipped")

            df = df[df['sites_per_hour'] >=3]
            
            # --- Calculate statistics per hour ---
            df['network_median_value'] = df.groupby(config_dict['timestamp_col'])[config_dict['value_col']].transform('median')
            df['network_typical_value'] = df.groupby(config_dict['timestamp_col'])['median_at_hour'].transform('median')
            df['network_median_z'] = df.groupby(config_dict['timestamp_col'])['z_score_mod'].transform('median')    
            df['local_z'] = df['z_score_mod'] - df['network_median_z'] # take delta from network z. local component

            # --- Flag regional events as hours where network median z exceeded threshold
            regional = df[df['network_median_z'] >= z_thresh].copy()
        
            # --- group these events to smooth out small timegaps
            regional = regional.sort_values('hours')
            regional_clusters = (regional['hours'].diff().fillna(0) > 24).cumsum() # differentiate unique regional events if more than 24 hours apart
            regional['event_ID'] = f'REG_{pollutant}_' + regional_clusters.astype(str)

            # --- create summary table from regional events
            regional_event_stats = regional.groupby('event_ID').agg(
                start_time = (config_dict['timestamp_col'],'min'),
                end_time = (config_dict['timestamp_col'],'max'),
                peak_value = ('network_median_value','max'),
                ).reset_index()
            regional_event_stats['event_scale'] = 'Regional'
            regional_event_stats['sites'] = 'Network-wide'
            regional_event_stats['typical_value_at_peak'] = regional.loc[regional.groupby('event_ID')['network_median_value'].idxmax(), 'network_typical_value'].values
            regional_event_stats[config_dict['pollutant_col']] = pollutant
            events_list.append(regional)
            summary_list.append(regional_event_stats)
            
        else:
            print(f"Median of {median_sites} sites per hour with {pollutant} data, cannot calculate regional stats."
                " Running clustering using individual site z-scores")

            # network stats are not meaningful with fewer than 3 sites
            df['network_median_value'] = np.nan
            df['network_typical_value'] = np.nan
            df['network_median_z'] = np.nan
            df['local_z'] = df['z_score_mod'] # in this case, the unadjusted z-score is used for clustering

        # check that there are at least some sites within the local_radius
        sites = df.drop_duplicates(subset = [config_dict['lon_col'],config_dict['lat_col']])
        gdf = gpd.GeoDataFrame(sites,geometry=gpd.points_from_xy(sites[config_dict['lon_col']], sites[config_dict['lat_col']]),crs="EPSG:4326")
        gdf = gdf.to_crs(gdf.estimate_utm_crs()) # localize
        nearest_idx, nearest_m = gdf.sindex.nearest(gdf.geometry, exclusive=True, return_all=False, return_distance=True)
        gdf['nearest_km'] = nearest_m / 1000
        within_local_distance = (gdf['nearest_km'] < local_distance_km).any()
        
        if not within_local_distance:
            warnings.warn(f"No sites in network are within defined local distance of {local_distance_km} km. No clusters will be identified",UserWarning)
                
        # --- Now, isolate local events and group them in space and time ---
        # --- Select events where the site local_z (adjusted by network median) exceeded the z threshold ---
        local = df[df['local_z'] >= z_thresh].sort_values(by=config_dict['timestamp_col'], ascending=True).copy()
            
        # --- cluster these events to identify spatiotemporal patterns    
        # --- Project data for more accurate and interpretable distances
        local_gdf = gpd.GeoDataFrame(local, geometry=gpd.points_from_xy(local[config_dict['lon_col']], local[config_dict['lat_col']]),crs="EPSG:4326").to_crs(gdf.estimate_utm_crs())
        local['x_km'] = local_gdf.geometry.x / 1000 # to km
        local['y_km'] = local_gdf.geometry.y / 1000
        eps_hours_local = 4 # 4 hour eps
        local['weighted_time_local'] = local['hours'] * local_distance_km / eps_hours_local # scale time eps to be equivalent to spatial eps 
        db_local = DBSCAN(eps=local_distance_km, min_samples=1).fit(local[['y_km','x_km', 'weighted_time_local']]) 
        local['event_ID'] = f'LOC_{pollutant}_' + db_local.labels_.astype(str) # assign ID for each unique event
        events_list.append(local)

        # --- create summary table from local events
        local_event_stats = local.groupby('event_ID').agg(
            start_time = (config_dict['timestamp_col'],'min'),
            end_time = (config_dict['timestamp_col'],'max'),
            sites = (config_dict['site_col'], lambda x: ', '.join(set(x.astype(str)))),
            peak_value = (config_dict['value_col'],'max')
            ).reset_index()
        local_event_stats['event_scale'] = 'Local'
        local_event_stats['typical_value_at_peak'] = local.loc[local.groupby('event_ID')[config_dict['value_col']].idxmax(), 'median_at_hour'].values
        local_event_stats[config_dict['pollutant_col']] = pollutant
        summary_list.append(local_event_stats)
        
        # --- create one table of events ---
        # --- listing ID's allows for multiple simultaneous local and regional events ---
        events = pd.concat(events_list)[[config_dict['site_col'],config_dict['timestamp_col'],'event_ID']]
        events = events.groupby([config_dict['site_col'],config_dict['timestamp_col']]).agg(event_ID = ('event_ID',list))
        
        # --- join event ID's back to disagg data
        df = df.merge(events, on=[config_dict['site_col'],config_dict['timestamp_col']], how='left')

        df_list_all.append(df)
        summary_list_all.append(pd.concat(summary_list, ignore_index=True)) 
                    
    # --- produce summary and disaggregated table for all pollutants ---
    summary = pd.concat(summary_list_all,ignore_index=True)
    df = pd.concat(df_list_all, ignore_index=True)

    # --- select columns to return ---
    df = df[[config_dict['site_col'],config_dict['timestamp_col'],config_dict['pollutant_col'],'median_at_hour','z_score_mod','local_z','network_median_value','network_median_z','network_typical_value','event_ID']]

    return(summary, df)