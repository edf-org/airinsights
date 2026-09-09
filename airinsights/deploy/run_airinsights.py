import pandas as pd
import airinsights as air

def run_airinsights(aqdata:pd.DataFrame,config_dict:dict):
    """Runs all Air Insights analytical methods sequentially and returns a dict of the output tables"""
    
    # start with measurement table
    aqdata = aqdata.copy()
    
    # run diurnal pattern script and add summary by time window of highest and lowest site
    diurnal_summary = air.diurnal_hotspots(aqdata,config_dict,z_thresh=1)
    site_avgs = diurnal_summary.groupby([config_dict['site_col'],'time_window'])['site_mean'].mean().round(1).reset_index() # ,config_dict['pollutant_col']
    g = site_avgs.groupby(['time_window'])['site_mean'] #config_dict['pollutant_col'], 
    extremes = pd.DataFrame({
        'lowest_avg':   g.min(),
        'highest_avg':  g.max(),
        'network_avg':  g.mean().round(1),
        'lowest_site':  site_avgs.loc[g.idxmin(), config_dict['site_col']].values,
        'highest_site': site_avgs.loc[g.idxmax(), config_dict['site_col']].values,
    }).reset_index()
    diurnal_summary = diurnal_summary.merge(extremes, on=['time_window'], how='left') # config_dict['pollutant_col'], 

    # run pollution event script
    events = air.pollution_event(aqdata,config_dict,verbose=True,window_size=60)
    events_df = aqdata.merge(events,on = [config_dict['site_col'],config_dict['timestamp_col'],config_dict['pollutant_col']],how="left") # join back to full df

    # run clustering script
    cluster_summary, cluster_df = air.classify_pollution_events(events_df,config_dict,z_thresh=2,local_distance_km=5)
    cluster_df = events_df.merge(cluster_df,on = [config_dict['site_col'],config_dict['timestamp_col'],config_dict['pollutant_col']],how="left") # join back to full df
    cluster_summary['duration'] = (cluster_summary['end_time'] - cluster_summary['start_time']).dt.total_seconds() / 3600 + 1

    # explode so one event per column (duplicates some timestamps)
    # necessary for going between summary and disagg in dashboard
    cluster_df = cluster_df.explode('event_ID')
    
    # add network description for current panel of dashboard
    cluster_df['network_description'] = pd.cut(cluster_df['network_median_z'],
                                               bins=[-float('inf'), -3, -2, -1, 1, 2, 3, float('inf')],
                                               labels=['extremely low', 'unusually low', 'somewhat low', 'similar',
                                                       'somewhat high', 'unusually high', 'extremely high'],)
    
    return {
        "diurnal_summary": diurnal_summary,
        "event_summary": cluster_summary,
        "meas_disagg": cluster_df,
    }