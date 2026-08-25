from ..download.get_openaq import get_openaq
from ..download.get_purpleair import get_purpleair
from google.api_core.exceptions import NotFound
import pandas as pd

def get_meas(config_dict,client):
    """Wrapper around data download scripts to download latest measurements for a specific data source"""
    # currently written for BQ only
    from google.cloud import bigquery

    # --- read existing ---
    try:
        existing = client.query(
            f"SELECT * FROM `{config_dict['meas_table']}`"
        ).to_dataframe()
        print("Existing measurements found")
    except NotFound:
        existing = pd.DataFrame()
        print("No existing measurements, starting fresh")

    # --- fetch new ---
    if config_dict['data_source'] == 'openaq':
        new_data = get_openaq(config_dict, existing)
    elif config_dict['data_source'] == 'purpleair':
        new_data = get_purpleair(config_dict, existing)
    else:
        raise ValueError(f"Unknown data_source {config_dict['data_source']!r}")

    # --- append to database ---
    if new_data.empty:
        print("No new data found, skipping upload")
        return existing

    client.load_table_from_dataframe(
        new_data, config_dict['meas_table'],
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()
    print(f"added {len(new_data)} rows to source data table")

    return new_data if existing.empty else pd.concat([existing, new_data], ignore_index=True)