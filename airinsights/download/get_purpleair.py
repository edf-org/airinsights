import requests
import time
import pandas as pd
from datetime import datetime, timezone, timedelta

def get_purpleair(config_dict,existing):

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
    week_ago = now - timedelta(days=7)
    
    def download_sensor(row):
        url = f"https://api.purpleair.com/v1/sensors/{row.sensor_index}/history"
        sensor_id = row.sensor_index
        end_date = int(now.timestamp())
        last_ts = last_seen.get(sensor_id) or week_ago
        start_date = int((last_ts + timedelta(hours=1)).timestamp())

        params = {
            "fields": 'pm2.5_alt',
            "start_timestamp": start_date, # start hour after the values we have
            "end_timestamp": end_date, # most recent complete hour
            "average": 60,
        }
        
        try:
            response = requests.get(url, headers={"X-API-Key": api_key}, params=params)
            response.raise_for_status()
            data = response.json()
            if not data.get("data"):
                return None
            temp = pd.DataFrame(data["data"], columns=data["fields"])
            temp["sensor_index"] = row.sensor_index
            temp["datetime"] = pd.to_datetime(temp["time_stamp"], unit="s", utc=True)
            new_rows = len(temp)
            print(f"added {new_rows} new hours for sensor {row.sensor_index}")
            return temp
        except Exception as e:
            print(f"skipping sensor {row.sensor_index}: {e}")
            return None
        finally:
            time.sleep(1)
            
    results = [res for res in map(download_sensor, purpleair_metadata.itertuples(index=False)) if res is not None]
    
    if results:
        df = pd.concat(results, ignore_index=True)
        df['pm2_5'] = df['pm2.5_alt'] * (3.4/3)
        df = df.merge(purpleair_metadata[['sensor_index','latitude','longitude']],on='sensor_index')
        return df[['sensor_index','datetime','latitude','longitude','pm2_5']]