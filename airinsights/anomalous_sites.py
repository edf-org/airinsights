import pandas as pd
import numpy as np

def get_hotspot_type(high_hours, time_bins):
    labels = [label for label, hours in time_bins.items() if high_hours.isin(hours).any()]
    if set(labels) == set(time_bins):
        return "All day"
    return ", ".join(labels) if labels else "None"

def diurnal_hotspots(
    input_data: pd.DataFrame, 
    config_dict: dict,
    z_thresh : int = 1,  
):
    
    # --- Read input data as df ---
    df = input_data.copy()

    # --- Extract coords to join back later ---
    coords = df[[config_dict['site_col'], config_dict['lat_col'], config_dict['lon_col']]].drop_duplicates()   
     
    # --- Add column for hour of day ---
    df["hour"] = df[config_dict['timestamp_col']].dt.hour

    # --- Timeframes to run stats on: 90 days and full dataset ---
    max_time = df[config_dict['timestamp_col']].max()
    timeframes = {
        "30d": df[df[config_dict['timestamp_col']] >= max_time - pd.Timedelta(days=30)],
        "90d": df[df[config_dict['timestamp_col']] >= max_time - pd.Timedelta(days=90)],
        "1y": df[df[config_dict['timestamp_col']] >= max_time - pd.DateOffset(years=1)],
        "all_time": df
    }

    # --- Time-of-day bins ---
    time_bins = {
        "Morning": list(range(6, 11)),
        "Midday": list(range(11, 17)),
        "Evening": list(range(17, 21)),
        "Night": list(range(21, 24)) + list(range(0, 6))
    }

    results = []

    # --- Calculate diurnal means for each site by timeframe ---
    for tf_name, df_tf in timeframes.items():

        if df_tf.empty:
            continue

        # --- Diurnal mean per sensor ---
        diurnal = (
            df_tf.groupby([config_dict['site_col'],"hour"])[config_dict['value_col']]
            .agg(site_mean="mean")
            .reindex(pd.MultiIndex.from_product( # Ensure all 24 hours per sensor
            [df_tf[config_dict['site_col']].unique().tolist(), range(24)],
            names=[config_dict['site_col'], "hour"]
            ))
            .reset_index()
        )

        # --- Network stats (across sensors at each hour) ---
        stats = (
            diurnal.groupby("hour")["site_mean"]
            .agg(
                network_median="median",
                network_mad=lambda x: np.nanmedian(np.abs(x - np.nanmedian(x)))
            )
            .reset_index()
        )
        diurnal = diurnal.merge(stats, on="hour", how="left")
        diurnal["network_mad"] = diurnal["network_mad"].replace(0, np.nan)

        # --- Calculate modified z-score for each site and hour ---
        
        diurnal["z_score_mod"] = (diurnal["site_mean"] - diurnal["network_median"]) / (1.4826 * diurnal["network_mad"])

        # --- Assign hotspot type by location ---
        
        for site, site_data in diurnal.groupby(config_dict['site_col']):

            elevated = site_data["z_score_mod"] > z_thresh
            hotspot = get_hotspot_type(site_data.loc[elevated, "hour"], time_bins)

            results.append(
                site_data.assign(
                    z_thresh=z_thresh,
                    elevated=elevated,
                    n_hours_elevated=elevated.sum(),
                    times_elevated=hotspot,
                    time_window=tf_name
                )
            )

    out = pd.concat(results, ignore_index=True)
    out = out.merge(coords,on=config_dict['site_col'],how='left') #bring back coords

    return out