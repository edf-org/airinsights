import pandas as pd
import airinsights as air

def run_airinsights(aqdata:pd.DataFrame,config_dict:dict):
    """Runs all Air Insights analytical methods sequentially and returns a dict of the output tables"""
    
    # start with measurement table
    aqdata = aqdata.copy()
    
    # run diurnal pattern script
    diurnal_summary = air.diurnal_hotspots(aqdata,config_dict,z_thresh=1)
                
    # run pollution event script
    events = air.pollution_event(aqdata,config_dict,verbose=True,window_size=60)
    events_df = aqdata.merge(events,on = [config_dict['site_col'],config_dict['timestamp_col'],config_dict['pollutant_col']],how="left") # join back to full df

    # run clustering script
    cluster_summary, cluster_df = air.classify_pollution_events(events_df,config_dict,z_thresh=2,local_distance_km=5)

    # explode so one event per column (duplicates some timestamps)
    # necessary for going between summary and disagg in dashboard
    cluster_df = cluster_df.explode('event_ID')
    
    return {
        "diurnal_summary": diurnal_summary,
        "event_summary": cluster_summary,
        "meas_disagg": cluster_df,
    }