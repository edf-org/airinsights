import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
 
# session that retries with backoff after errors
session = requests.Session()
session.mount("https://", HTTPAdapter(pool_maxsize=10,max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))

def get_purpleair(config_dict,existing):
    """ Retrieves historical and recent purpleair data for a sensor 'group' defined in the config. Uses ALT CF=3.4 calibration factor for PM2.5. """
    
    if existing is None or existing.empty:
        last_seen = {}
    else:
        last_seen = existing.groupby('sensor_index')[config_dict['timestamp_col']].max().to_dict()

    # get params from config
    group_number = config_dict['group_number']
    api_key = config_dict['api_key']

    # get locations and sensors
    url = f"https://api.purpleair.com/v1/groups/{group_number}/members"
    response = requests.get(url, headers={"X-API-Key": api_key},params={"fields": "name,longitude,latitude,last_seen"})
    data = response.json()
    purpleair_metadata = pd.DataFrame(data["data"], columns=data["fields"])

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    backfill = now - timedelta(days=365) # backfill to a year ago
    
    # fetch sensors in parallel
    def run(row):
        return _purpleair_sensor(row.sensor_index, api_key, last_seen, backfill, now)

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = [df for df in pool.map(run, purpleair_metadata.itertuples(index=False))
                   if df is not None]
        
    if results:
        df = pd.concat(results, ignore_index=True)
        df['pm2_5'] = df['pm2.5_alt|3.4']
        df = df.merge(purpleair_metadata[['sensor_index','latitude','longitude']],on='sensor_index')
        return df[['sensor_index','datetime','latitude','longitude','pm2_5']]
    
def _purpleair_chunk(sensor_id, start_date, end_date, api_key):
    """ Get a 14-day chunk of hourly purpleair data (max allowed by API)"""
    
    url = f"https://api.purpleair.com/v1/sensors/{sensor_id}/history"
    params = {
        "fields": 'pm2.5_alt|3.4',
        "start_timestamp": int(start_date.timestamp()),
        "end_timestamp": int(end_date.timestamp()),
        "average": 60,
    }
    try:
        response = session.get(url, headers={"X-API-Key": api_key}, params=params)
        response.raise_for_status()
        data = response.json()
        if not data.get("data"):
            return None
        temp = pd.DataFrame(data["data"], columns=data["fields"])
        temp["sensor_index"] = sensor_id
        temp["datetime"] = pd.to_datetime(temp["time_stamp"], unit="s", utc=True)
        return temp
    except requests.exceptions.RequestException as e: 
        print(f"  skipping chunk {start_date:%Y-%m-%d}->{end_date:%Y-%m-%d} for sensor {sensor_id}: {e}")
        return None

def _purpleair_sensor(sensor_id, api_key, last_seen, backfill, now):
    """ Fetch all new hours for one sensor in 14-day chunks """
    # start the hour after data we already have; otherwise go back the full backfill window
    last_ts = last_seen.get(sensor_id)
    if last_ts is None or pd.isna(last_ts):
        window_start = backfill
    else:
        window_start = last_ts + timedelta(hours=1)
 
    # download data in chunks of 14 days or less
    data = []
    chunk_start = window_start
    while chunk_start < now:
        chunk_end = min(chunk_start + timedelta(days=14), now)
        chunk = _purpleair_chunk(sensor_id, chunk_start, chunk_end, api_key)
        if chunk is not None:
            data.append(chunk)
        chunk_start = chunk_end  # next window begins where this one ended
 
    if not data:
        return None
    
    sensor_df = pd.concat(data, ignore_index=True).drop_duplicates(subset="datetime")
    print(f"added {len(sensor_df)} new hours for sensor {sensor_id}")
    return sensor_df

def _purpleair_credits(api_key):
    r = requests.get("https://api.purpleair.com/v1/organization",
                     headers={"X-API-Key": api_key})
    r.raise_for_status()
    return r.json().get("remaining_points")