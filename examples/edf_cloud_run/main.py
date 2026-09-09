import airinsights as air
from google.cloud import bigquery
import pandas as pd
from pathlib import Path
from google.api_core.exceptions import NotFound
import os
import sys

# add a config file to the config folder to add to routine
config_dir = Path(__file__).parent / "config"

# pipeline that runs per config file in parallel
def city_pipeline(config_path):
    # --- Load measurement file and config, identify last updated time for each site --- 
    # --- Run measurement pipeline for data source specified in config and update database ---
    config = air.load_config(config_path)
    client = bigquery.Client('edf-aq-data')
    print("Checking for existing measurements")
    df = air.get_meas(config,client)
    
    # --- Localize timestamp ----
    # this should go in helpers 
    df[config['timestamp_col']] = df[config['timestamp_col']].dt.tz_convert(config['local_tz'])
    # remove tz for bigquery datetime with .tz_localize(None)
    
    # --- Run full set of air insights methods ---
    print("Running air insights methods")
    results_dict = air.run_airinsights(df,config)
    
    # --- Get any missing airtracker trajectories for local pollution events ---
    try:
        existing_events = client.query(f"SELECT DISTINCT {config['site_col']},{config['timestamp_col']} FROM `{config['airtracker_table']}`").to_dataframe()
    except NotFound as e:
        existing_events = pd.DataFrame(columns=[config['site_col'], config['timestamp_col']])

    # select only new local events - only go back to 2025
    disagg = results_dict["meas_disagg"]
    local_events = disagg[disagg['event_ID'].str.contains('LOC', na=False) 
                            & (disagg[config['timestamp_col']] >= pd.Timestamp('2025-01-01', tz=config['local_tz']))]
    local_events = local_events[
        ~local_events.set_index([config['site_col'], config['timestamp_col']]). #local events cannot be in existing events from BQ
        index.isin(existing_events.set_index([config['site_col'], config['timestamp_col']]).index)
    ]
    
    airtracker = air.get_airtracker(local_events,config) 
        
    # --- Write results to databse ---
    job_config_append = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job_config_replace = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    
    client.load_table_from_dataframe(results_dict["meas_disagg"],config['dataset'] + '.' + config['data_source'] + '_disagg',job_config=job_config_replace).result() 
    print("overwrote disagg table")

    client.load_table_from_dataframe(results_dict["event_summary"],config['dataset']  + '.' + config['data_source'] + '_cluster',job_config=job_config_replace).result() 
    print("overwrote event table")

    client.load_table_from_dataframe(results_dict["diurnal_summary"],config['dataset']  + '.' + config['data_source'] + '_diurnal',job_config=job_config_replace).result() 
    print("overwrote diurnal table")
    
    client.load_table_from_dataframe(airtracker,config['airtracker_table'],job_config=job_config_append).result() 
    print(f"added {len(airtracker)} rows to airtracker table")

    # --- Update airtracker event_ID based on new clusters that can evolve over time ---
    disagg_table = config['dataset'] + '.' + config['data_source'] + '_disagg'
    merge_sql = f"""
    MERGE `{config['airtracker_table']}` airtracker
    USING (SELECT {config['site_col']}, {config['timestamp_col']},ANY_VALUE(event_ID) AS event_ID
            FROM `{disagg_table}`
            WHERE event_ID LIKE 'LOC%'
            GROUP BY {config['site_col']}, {config['timestamp_col']}) clusters
    ON  airtracker.{config['site_col']} = clusters.{config['site_col']}
    AND airtracker.{config['timestamp_col']} = clusters.{config['timestamp_col']}
    WHEN MATCHED THEN
      UPDATE SET event_ID = clusters.event_ID
    """
    client.query(merge_sql).result()
    print("refreshed event_IDs in airtracker table")
    
# function that triggers city_pipeline for each config using input from Cloud Run Jobs
def main(): 
    configs = sorted(config_dir.glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No configs found in {config_dir.resolve()}")
    
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", 0)) # when not provided task from GCP, runs 0 as default

    if task_index >= len(configs):
        print(f"Task {task_index}: only {len(configs)} configs exist, exiting cleanly")
        sys.exit(0)

    config_path = configs[task_index]
    print(f"Task {task_index}: running {config_path.name}")
    city_pipeline(config_path)
    
if __name__ == "__main__":
    main()