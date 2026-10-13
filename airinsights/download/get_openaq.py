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
import os

def get_openaq(config_dict:dict,last_seen:dict,years_back: int = 5,reference_only: bool = False):
    """Fetch OpenAQ air quality data (historical backfill + recent measurements) for sites within a bounding box.

    Parameters
    ----------
    config_dict: dict
        Config dict from air.load_config that includes 'bounding_box' for openaq download
    last_seen: dict
        Last stored timestamp per location_id and parameter.
        If None/empty, triggers a full historical backfill from the S3 archive.
    years_back: int
        How many years back to backfill data (defaults to 5)
    reference_only: bool
        If true, only download data for reference monitors in bounding box. Defaults to false.
    Requires the OPENAQ_KEY environment variable.

    Returns:
        DataFrame of downloaded measurements. site_name is the location name followed by the location_id
        in parentheses, so it is unique.
    """
    last_seen = last_seen or {}
    
    # get params from config
    if 'bounding_box' not in config_dict:
        raise KeyError("Config dict missing 'bounding_box' for area to download")
    bbox = tuple(config_dict['bounding_box'])
    pollutants = [p.lower() for p in config_dict['pollutant_map'].values()] # parameters to query
    
    # get api key from environment
    api_key = os.environ.get("OPENAQ_KEY")
    if not api_key:
        raise KeyError(
            "Environment variable 'OPENAQ_KEY' not found. Please set it in your environment."
        )
        
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openaq-data-archive'
    client = OpenAQ(api_key=api_key)

    # get locations and sensors
    locations = client.locations.list(bbox=bbox, limit=1000).results
    if reference_only: # optionally filter for reference monitors only
        locations = [loc for loc in locations if loc.is_monitor]
    
    sensors = [
        (loc, sensor)
        for loc in locations
        for sensor in loc.sensors
        if sensor.parameter.name.lower() in pollutants
    ]
    print(f"{len(sensors)} data streams found at {len(locations)} sites")

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
    
    # site names can repeat across locations, so add the location id (also fills blank names)
    metadata["site_name"] = metadata["site_name"].fillna("Unnamed") + " (" + metadata["location_id"].astype(str) + ")"
    location_ids = {loc.id for loc, sensor in sensors} # only unique locations to loop over
    cols = ['datetime','parameter','value','units','sensors_id','location_id','lat','lon']

    # if no historical exists, backfill
    if not last_seen:
        print("Fetching historical data")
        cutoff = datetime.now(timezone.utc) - timedelta(days=365 * years_back)
        years = range(cutoff.year, datetime.now(timezone.utc).year + 1)
        
        # list files
        files = [
            obj['Key']
            for loc_id in location_ids
            for year in years
            for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket,
                                                                     Prefix=f'records/csv.gz/locationid={loc_id}/year={year}/')
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
        with ThreadPoolExecutor(max_workers=10) as ex:
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
        print("Fetching recent measurements")
        key = (loc.id, sensor.parameter.name)
        end_date = last_seen.get(key) or week_ago
    
        # page through all new measurements (1000 per page) so none are cut off
        rows, page = [], 1
        while True:
            measurements = client.measurements.list(
                sensors_id=sensor.id,
                data='measurements',
                datetime_from=(end_date + timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                datetime_to=now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                page=page,
                limit=1000,
            )
            rows += measurements.results
            if len(measurements.results) < 1000: # last page
                break
            page += 1

        if not rows:
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
        } for r in rows]))
        
        if measurements.headers.x_ratelimit_remaining <= 5:
                print(f"Rate limit nearly exhausted, waiting {measurements.headers.x_ratelimit_reset}s...")
                time.sleep(measurements.headers.x_ratelimit_reset)
    
    dfs = [df[cols] for df in [df_hist, pd.concat(results) if results else pd.DataFrame()] if not df.empty]
    
    if not dfs:
        return pd.DataFrame()
    
    out = pd.concat(dfs)
    out = out.merge(metadata, on='location_id', how='left')
    return out