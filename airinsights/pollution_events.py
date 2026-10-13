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

# ===============================================================
# Pollution event detection and local vs. regional classification using median absolute deviation (MAD) and DBSCAN clustering
# ===============================================================

# --- Import packages ---
import pandas as pd
import numpy as np
import polars as pl
from airinsights.helpers import _validate_hourly
from sklearn.cluster import DBSCAN
import geopandas as gpd
import warnings

def pollution_events(input_data: pd.DataFrame,
                     config_dict: dict,
                     window_size : int = 60,
                     z_thresh : float = 2,
                     local_distance_km : float = 1
                     ):
    """Flags elevated air pollution measurements and groups into 'Regional', 'Local', or 'Unknown' events using clustering

    Elevated measurements are flagged by calculating the median, median absolute deviation (MAD), and modified Z-score 
    for each monitor by hour of day to compare individual monitor observations to the monitor's historic observations at each hour of day. 
    Measurements are log transformed for a more normal distribution and calculations are done over a user-defined (default is 60 days) rolling window, 
    between 30 days and 365 days, with a minimum of 75% of data required. 
    
    * 'Regional' (i.e. the extent of the network) events are identified when the network median z-score (network Z) 
    matches or exceeds the defined Z threshold (default is 2). Hours less than 24 hours apart are grouped into the same event.
    * 'Local' events are identified when the local Z (the difference between the site Z and the network Z) matches or exceeds 
    the defined Z threshold. For these events, 3-dimensional DBSCAN is tuned with a defined spatial eps (default: 1 km) 
    and a temporal eps of 4 hours. This groups local events that are close enough in space and time that they could be associated 
    with the same local emissions event.
    * 'Unknown' events are labeled when only one monitor has a valid Z-score, meaning it is not possible to distinguish 
    between local and regional scale.

    In some cases, the method will identify 'Regional' and 'Local' events simultaneously. This occurs when
    the network z-score is above the z_thresh (e.g. network_z = 2.5) but an individual site is still
    enhanced by more than the z_thresh relative to the network (e.g. at site E, site_z = 6 so local_z = 3.5).

    See examples/pollution_events.ipynb on GitHub for a full working example: 
    https://github.com/edf-org/airinsights/blob/main/examples/pollution_events.ipynb

    Parameters
    ----------
    input_data: pd.DataFrame
        A pandas DataFrame containing AQ data that was read using one of the helpers.read_aqdata_[x] routines.
    config_dict: dict
        A dictionary containing input_data parameters that was read using one of the helpers.read_aqdata_[x] routines.
    window_size : int, default 60
        Number of days in the rolling window used to determine 'typical' pollutant values and calculate z-scores. 
    z_thresh : float, default 2
        Threshold for modified z-score to include measurements in pollution event classification. Default value of 2 ("Unusually high").
    local_distance_km : float, default 1
        Eps (in km) for local DBSCAN run; the greatest distance at which monitors are grouped as neighbors. Default value of 1 km.

    Returns
    -------
    events : pd.DataFrame
        A pandas DataFrame with one row per classified pollution event, with the following columns:

            **event_ID**: unique identifier for the event, prefixed by scale and pollutant (e.g. "REG_PM2.5_0", "LOC_PM2.5_3")

            **pollutant**: pollutant of the event

            **start_time**, **end_time**: start and end times of the pollution event

            **duration_hours**: number of hours from start to end of the event

            **scale**: "Regional", "Local", or "Unknown"

            **sites**: "Network-wide" for Regional events, or a comma-separated list of sites for Local and Unknown events

            **peak_value**: maximum measured value observed during the event

            **typical_at_peak**: the network's (Regional) or site's (Local or Unknown) typical value for the peak's hour of day, for comparison against peak_value

    measurements: pd.DataFrame
        A pandas DataFrame containing the site, timestamp, and pollutant columns from the input data with the following columns appended:

            **hour**: hour of day

            **days_captured**: number of days with data captured in the historical window

            **site_typical**: measurement median for hour of day over historical window

            **site_z**: modified Z-score of the AQ measurement calculated using MAD over historical window

            **elevated_flag**: severity of elevated measurement based on site_z. A Z-score greater than 2 is "Unusually high", 
            a Z-score greater than 3 is "Extremely high". Z-scores below 2 are returned as NULL.  
            "Insufficient number of days captured" is returned if window size was insufficient to calculate a Z-score

            **n_sites**: number of monitoring sites with a valid z_score at that timestamp (used for network_z)

            **network_z**: network-wide median z-score at that timestamp

            **local_z**: the site's z-score, adjusted for the network z-score (local_z = site_z - network_z)

            **network_value**: network-wide median measured value at that timestamp

            **network_typical**: network-wide median of each site's typical value for that hour of day

            **event_ID**: list of pollution event_ID's associated with that site/timestamp/pollutant

    Notes
    -----
    The function removes values <= 0 to allow for log-transformation of data.
    
    The modified Z-score is calculated as ``abs[(hourly_value - median_value) / 1.4826 * MAD]``, where 1.4826 is a 
    scaling factor used to make MAD comparable to standard deviation. See https://en.wikipedia.org/wiki/Median_absolute_deviation
    
    """

    # --- Run MAD method to flag elevated measurements and join back metadata ---
    flagged = _flag_elevated_measurements(input_data, config_dict, window_size=window_size)
    join_cols = [config_dict['site_col'], config_dict['timestamp_col'], config_dict['pollutant_col']]
    df = flagged.merge(
        input_data[join_cols + [config_dict['value_col'], config_dict['lat_col'], config_dict['lon_col']]],
        on=join_cols,
        how='left'
    )
    
    # --- Drop sites with missing coordinates (required for local clustering) and warn ---
    missing_coords = df[config_dict['lat_col']].isna() | df[config_dict['lon_col']].isna()
    if missing_coords.any():
        missing_sites = df.loc[missing_coords, config_dict['site_col']].unique().tolist()
        warnings.warn(f"Excluded rows with missing coordinates for sites: {missing_sites}")
        df = df[~missing_coords]

    # --- Calculate network statistics for each hour, grouped by pollutant ---
    grouped = df.groupby([config_dict['timestamp_col'],config_dict['pollutant_col']])
    df['network_value'] = grouped[config_dict['value_col']].transform('median') # network median measurement value 
    df['network_typical'] = grouped['site_typical'].transform('median') # network median of site typical values (their historical medians)
    df['network_z'] = grouped['site_z'].transform('median') # network median z score
    df['local_z'] = df['site_z'] - df['network_z'] # take delta from network z to get local component
    df['n_sites'] = grouped['site_z'].transform('count')
    # hours from start is used in clustering later
    df['hours'] = (df[config_dict['timestamp_col']] - df[config_dict['timestamp_col']].min()).dt.total_seconds() / 3600 
    # use metadata to determine appropriate projection
    all_sites = df.drop_duplicates(subset = [config_dict['lon_col'],config_dict['lat_col']])
    utm = gpd.GeoDataFrame(all_sites, geometry=gpd.points_from_xy(all_sites[config_dict['lon_col']], 
                                                              all_sites[config_dict['lat_col']]), 
                                                              crs="EPSG:4326").estimate_utm_crs()

    # --- Select regional events as hours where network median z matched or exceeded threshold
    regional = df[df['network_z'] >= z_thresh].sort_values(config_dict['timestamp_col'])

    # --- Select local events as hours where local z matched or exceeded threshold
    local = df[df['local_z'] >= z_thresh].sort_values(config_dict['timestamp_col'])
    # prepare for clustering by projecting coords, weighting the temporal dimension
    local_gdf = gpd.GeoDataFrame(local, geometry=gpd.points_from_xy(local[config_dict['lon_col']], local[config_dict['lat_col']]),crs="EPSG:4326").to_crs(utm)
    local['x_km'] = local_gdf.geometry.x / 1000 # to km
    local['y_km'] = local_gdf.geometry.y / 1000
    eps_hours_local = 4 # 4 hour eps
    local['weighted_time_local'] = local['hours'] * local_distance_km / eps_hours_local # scale time eps to be equivalent to spatial eps 
    
    # --- Loop through pollutants to assign event ID's and cluster them
    regional['event_ID'], local['event_ID'] = None, None
    for pollutant in df[config_dict['pollutant_col']].unique():

        # Check how many monitors and whether they are within local clustering distance
        sites = df[df[config_dict['pollutant_col']] == pollutant].drop_duplicates(subset=[config_dict['lon_col'], config_dict['lat_col']])
        sites_gdf = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites[config_dict['lon_col']], sites[config_dict['lat_col']]), crs="EPSG:4326").to_crs(utm)
        if len(sites_gdf) == 1:
            warnings.warn(f"Only one site measuring {pollutant}. Local and regional events cannot be distinguished and will be labelled 'Unknown'.")
        else:
            _, nearest_m = sites_gdf.sindex.nearest(sites_gdf.geometry, exclusive=True, return_all=False, return_distance=True)
            if not (nearest_m / 1000 < local_distance_km).any():
                warnings.warn(f"No sites measuring {pollutant} are within defined local distance of {local_distance_km} km. Multi-site local events will not be identified.")    

        # Regional events: group if <=24 hours apart
        reg = regional[regional[config_dict['pollutant_col']] == pollutant]
        reg_clusters = (reg['hours'].diff().fillna(0) > 24).cumsum() # differentiate unique regional events if more than 24 hours apart
        reg_labels = f'REG_{pollutant}_' + reg_clusters.astype(str)
        regional.loc[reg.index, 'event_ID'] = reg_labels # assign labels back to df

        # Local events: cluster nearby sites and hours with DBSCAN
        loc = local[local[config_dict['pollutant_col']] == pollutant]
        if not loc.empty: # if there are any local events
            # min_samples=1 is unusual, but we use it here because we have already established that every record in this 
            # dataset is a pollution event, and the purpose of the clustering is to find neighboring sites and times if they exist
            db_loc = DBSCAN(eps=local_distance_km, min_samples=1).fit(loc[['y_km','x_km', 'weighted_time_local']])
            loc_labels = f'LOC_{pollutant}_' + pd.Series(db_loc.labels_, index=loc.index).astype(str) 
            local.loc[loc.index, 'event_ID'] = loc_labels # assign labels back to df

    # Unknown events: if no network (n<2), classify as unknown since scale cannot be determined
    mask = regional.groupby('event_ID')['n_sites'].transform('max') < 2
    unknown = regional[mask & regional['site_z'].notna()].copy()
    unknown['event_ID'] = unknown['event_ID'].str.replace(r'^REG_', 'UNK_', regex=True)
    regional = regional[~mask] # remove from regional events, keep separate

    # --- combine data for all events into df
    all_events = pd.concat([
        regional.assign(scale='Regional',
                        event_value=regional['network_value'], 
                        event_typical=regional['network_typical']),
        unknown.assign(scale='Unknown',
                       event_value=unknown[config_dict['value_col']],
                       event_typical=unknown['site_typical']),
        local.assign(scale='Local', 
                     event_value=local[config_dict['value_col']], 
                     event_typical=local['site_typical'])
    ], ignore_index=True)

    # --- create summary table from all events
    grouped = all_events.groupby(['event_ID', config_dict['pollutant_col'],'scale'])
    events = grouped.agg(
        start_time = (config_dict['timestamp_col'], 'min'),
        end_time = (config_dict['timestamp_col'], 'max'),
        sites = (config_dict['site_col'], lambda x: ', '.join(sorted(set(x.astype(str))))),
        peak_value = ('event_value', 'max'),
    ).reset_index()
    events['typical_at_peak'] = all_events.loc[grouped['event_value'].idxmax(), 'event_typical'].values
    events['duration_hours'] = (events['end_time'] - events['start_time']).dt.total_seconds() / 3600 + 1

    # instead of listing individual sites for regional event, they are 'network-wide'
    events.loc[events['scale'] == 'Regional', 'sites'] = 'Network-wide'

    # join ID's back to measurements
    event_ids = all_events.groupby(join_cols)['event_ID'].agg(list).reset_index()
    measurements = df.merge(event_ids, on=join_cols, how='left')

    # --- select columns to return
    measurements = measurements[join_cols + [
        'hour',
        'days_captured',
        'site_typical',
        'site_z',
        'elevated_flag',
        'n_sites',
        'network_z',
        'local_z',
        'network_value',
        'network_typical',
        'event_ID'
    ]]

    events = events[[
        'event_ID',
        config_dict['pollutant_col'],
        'start_time',
        'end_time',
        'duration_hours',
        'scale',
        'sites',
        'peak_value',
        'typical_at_peak'
    ]]

    return events, measurements
        
def _flag_elevated_measurements(input_data : pd.DataFrame,
                                config_dict : dict,
                                window_size : int = 60
                                ):
    """Identifies and flags elevated measurements in air quality timeseries

    This function calculates the median, median absolute deviation (MAD), and modified Z-score for each monitor by hour of day to compare individual monitor
    observations to the monitor's historic observations at each hour of day. Measurements are log transformed for a more normal distribution and calculations
    are done over a user-defined (default is 60 days) rolling window, between 30 days and 365 days, with a minimum of 75% of data required. 
    
    """
    # --- Read input data as df ---
    df = input_data.copy()

    # --- Validate hourly time resolution ---
    df = _validate_hourly(df, config_dict)

    # --- Parse window_size argument
    if not isinstance(window_size, (int, np.integer)):
        raise TypeError(f"integer expected, got {type(window_size).__name__}")
    elif window_size < 30 or window_size > 365:
        raise ValueError("Window size must be between 30 and 365.")

    min_days_in_window = round(window_size * 0.75)

    # --- Add column for hour of day ---
    df["hour"] = df[config_dict['timestamp_col']].dt.hour
    
    # --- Drop NA's and filter for values > 0 ---
    df = df.dropna(subset=[config_dict['site_col'], config_dict['value_col']])  # drop NA from necessary columns
    df = df[df[config_dict['value_col']] > 0]

    # --- Log-transform measurements
    df['value_log'] = np.log(df[config_dict['value_col']]) # previous analysis shows log-transformed measurements more closely match normal distribution; future work could make this optional

    #--- Compute diurnal (hourly) medians and MAD per monitor ---
    #--- Polar package used to optimize for speed over pandas .rolling.agg
    # polars throwing bug with local tz's. convert to UTC and back later
    tz = df[config_dict['timestamp_col']].dt.tz
    df[config_dict['timestamp_col']] = df[config_dict['timestamp_col']].dt.tz_convert("UTC")

    MAD = (
        pl.from_pandas(df).sort(config_dict['timestamp_col'])
        .rolling( 
            index_column=config_dict['timestamp_col'],
            period=f"{window_size}d",
            group_by=[config_dict['site_col'],config_dict['pollutant_col'], "hour"]
        ).agg(
            median = pl.col('value_log').median(),
            MAD = (pl.col('value_log') - pl.col('value_log').median()).abs().median(),
            days_captured = pl.len()
        ).to_pandas()) 

    # filter out values with insufficient window size
    window_mask = MAD["days_captured"] < min_days_in_window
    MAD.loc[window_mask, ["median", "MAD"]] = np.nan

    # --- Join back to other columns ---
    # --- Compute z-scores (z-score mod for MAD using scalar) and classify event (if >= 3 it is 'extreme', if >= 2 it is 'unusual') ---
    df = df.merge(MAD,on=[config_dict['site_col'],config_dict['timestamp_col'],config_dict['pollutant_col'],"hour"],how="left")
    df["z_score_mod"] = np.where(df["MAD"] > 0, # don't calculate if MAD is zero
                                 (df["value_log"] - df["median"]) / (1.4826 * df["MAD"]),
                                 np.nan)
    df["elevated_flag"] = np.select([df["z_score_mod"].isna(), df["z_score_mod"] >= 3, df["z_score_mod"] >= 2],
              ["Insufficient number of days captured", "Extremely high", "Unusually high"], None)
    df[config_dict['timestamp_col']] = df[config_dict['timestamp_col']].dt.tz_convert(tz)   # convert back to tz
    
    # --- Transform median back to concentration space ---
    df["median"] = np.exp(df["median"])
    
    # --- Select columns to return ---
    out = (df[[config_dict['site_col'],
               config_dict['timestamp_col'],
               config_dict['pollutant_col'],
               'z_score_mod',
               'elevated_flag',
               'hour',
               'median',
               'days_captured']].
               rename(columns={"median":"site_typical","z_score_mod":"site_z"})
    )
    
    return(out)