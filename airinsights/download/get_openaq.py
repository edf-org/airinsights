from openaq import OpenAQ
import pandas as pd
from datetime import datetime, timedelta, timezone
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pyarrow as pa
import pyarrow.csv as pac
import pyarrow.compute as pc
import io
import gzip
from concurrent.futures import ThreadPoolExecutor
import time

def get_openaq(config_dict,existing):
    
    if existing is None or existing.empty:
        last_seen = {}
    else:
        last_seen = existing.groupby(['location_id', config_dict['pollutant_col']])[config_dict['timestamp_col']].max().to_dict()
        
    # get params from config
    bbox = tuple(config_dict['bounding_box'])
    api_key = config_dict['openaq_api_key']
    pollutants = [p['name'].lower() for p in config_dict['pollutants'].values()] # nested with pollutant names and units
    
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openaq-data-archive'
    client = OpenAQ(api_key=api_key)

    # get locations and sensors
    locations = client.locations.list(bbox=bbox, limit=1000)
    sensors = [
        (loc, sensor)
        for loc in locations.results
        for sensor in loc.sensors
        if sensor.parameter.name.lower() in pollutants
    ]
    print(f"{len(sensors)} data streams found at {len(locations.results)} sites")

    # save metadata to join later
    metadata = pd.DataFrame([
    {
        'location_id':   loc.id,
        'site_name':     loc.name,
        'site_locality': loc.locality,
        'provider':      loc.provider.name,
        'monitor_type':  'reference' if loc.is_monitor else 'low-cost',
    }
    for loc, sensor in sensors]).drop_duplicates('location_id')
    
    location_ids = {loc.id for loc, sensor in sensors} # only unique locations to loop over
    cols = ['datetime','parameter','value','units','sensors_id','location_id','lat','lon']

    # if no historical exists, backfill
    if not last_seen:
        # list files
        files = [
            obj['Key']
            for loc_id in location_ids
            for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=f'records/csv.gz/locationid={loc_id}/')
            for obj in page.get('Contents', [])
        ]
        # decompress and read
        def fetch(file):
            body = s3.get_object(Bucket=bucket, Key=file)['Body'].read()
            table = pac.read_csv(io.BytesIO(gzip.decompress(body)))
            if 'parameter' not in table.schema.names:
                return None
            return table.filter(pc.is_in(table['parameter'], value_set=pa.array(pollutants))) # filter for pollutants of interest

        # this takes about 7-10s per thousand files
        with ThreadPoolExecutor(max_workers=20) as ex:
            tables = [t for t in ex.map(fetch, files) if t is not None and t.num_rows > 0]
        
        df_hist = pa.concat_tables([t.select([t.schema.names.index(c) for c in cols]) for t in tables]).to_pandas()
        df_hist['datetime'] = pd.to_datetime(df_hist['datetime'], utc=True) # this is in utc
        print(f"Historical: {len(df_hist)} rows downloaded")
    else:
        df_hist = pd.DataFrame()
        print("Historical already exists, skipping to recent measurements")
    
    # real-time
    # get last seen either from existing data or from newly downloaded historical
    # don't look more than 7 days back for cost
    if not last_seen:
        last_seen = df_hist.groupby(['location_id', 'parameter'])[config_dict['timestamp_col']].max().to_dict()

    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    results = []
    for loc, sensor in sensors:
        key = (loc.id, sensor.parameter.name)
        end_date = last_seen.get(key) or week_ago
    
        measurements = client.measurements.list(
            sensors_id=sensor.id,
            data='measurements',
            datetime_from=(end_date + timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ'), #don't look further than 7 days back
            datetime_to=now.strftime('%Y-%m-%dT%H:%M:%SZ'),
            limit=1000,
        )
                
        if not measurements.results:
            continue # if no new data, skip to next sensor
        
        results.append(pd.DataFrame([{
            'datetime': pd.Timestamp(r.period.datetime_to.utc, tz='UTC'),
            'parameter': r.parameter.name,
            'value': r.value,
            'units': r.parameter.units,
            'sensors_id': sensor.id,
            'location_id': loc.id,
            'lat': loc.coordinates.latitude,
            'lon': loc.coordinates.longitude,
        } for r in measurements.results]))
        
        if measurements.headers.x_ratelimit_remaining <= 5:
                print(f"Rate limit nearly exhausted, waiting {measurements.headers.x_ratelimit_reset}s...")
                time.sleep(measurements.headers.x_ratelimit_reset)
    
    dfs = [df[cols] for df in [df_hist, pd.concat(results) if results else pd.DataFrame()] if not df.empty]
    
    if not dfs:
        return pd.DataFrame()
    
    out = pd.concat(dfs)
    out = out.merge(metadata, on='location_id', how='left')
    return out