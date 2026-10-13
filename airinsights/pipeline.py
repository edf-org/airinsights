from pathlib import Path
import dlt
import pandas as pd
import airinsights as air
from airinsights.download import get_openaq, get_purpleair, get_esdr
from airinsights.helpers import _dedupe, _localize_tz, _melt_long

def run_airinsights(aqdata:pd.DataFrame,config_dict:dict):
    """Runs all Air Insights analytical methods sequentially and returns a dict of the output tables"""
    
    # start with measurement table
    aqdata = aqdata.copy()
    site, ts, pol = config_dict['site_col'], config_dict['timestamp_col'], config_dict['pollutant_col']

    # run anomalous sites script and add summary by time window of highest and lowest site
    anomalous = air.anomalous_sites(aqdata,config_dict,z_thresh=1)
    g = ['timeframe', pol] # summarize by timeframe and pollutant
    site_avgs = anomalous.groupby(g + [site])['site_mean'].mean().round(1).dropna().reset_index() 
    g_means = site_avgs.groupby(g)['site_mean'] 
    extremes = pd.DataFrame({
        'lowest_avg':   g_means.min(),
        'highest_avg':  g_means.max(),
        'network_avg':  g_means.mean().round(1),
        'lowest_site':  site_avgs.loc[g_means.idxmin(), site].values,
        'highest_site': site_avgs.loc[g_means.idxmax(), site].values,
    }).reset_index()
    anomalous = anomalous.merge(extremes, on=g, how='left')

    # run pollution events script
    events, measurements = air.pollution_events(aqdata,config_dict,window_size=60,z_thresh=2,local_distance_km = 5)
    measurements = aqdata.merge(measurements,on=[site, ts, pol],how="left") # join back to full df
    
    # explode so one event per column (duplicates some timestamps)
    # necessary for going between summary and disagg in dashboard
    measurements = measurements.explode('event_ID')
    
    # add network description for current panel of dashboard
    # todo - add to pollution_events func
    measurements['network_description'] = pd.cut(measurements['network_z'],
                                               bins=[-float('inf'), -3, -2, -1, 1, 2, 3, float('inf')],
                                               labels=['extremely low', 'unusually low', 'somewhat low', 'similar',
                                                       'somewhat high', 'unusually high', 'extremely high'],)

    # run long-term trends (needs at least 3 years of data)
    try:
        trends_network, trends, trends_by_month, trends_data = air.annual_trends(aqdata, config_dict)
    except ValueError as e:
        print(f"Skipping trends: {e}")
        trends_network = trends = trends_by_month = trends_data = pd.DataFrame()
        trends_data = {}
    
    return {
            "anomalous_sites": anomalous,
            "pollution_events": events,
            "meas_disagg": measurements,
            "annual_trends": trends,
            "network_trends": trends_network,
            "trends_by_month": trends_by_month,
            "trends_network_data": trends_data.get('network_monthly', pd.DataFrame()),
            "trends_site_data": trends_data.get('site_monthly', pd.DataFrame()),
        }

dlt.config['schema.naming'] = 'sql_cs_v1' # keep column names as they are

def _resolve_storage_destination(storage_type, location):
    """resolve a dlt destination, dataset name and location for a storage type"""

    # validate data storage options
    if storage_type not in ('parquet', 'csv', 'sqlite', 'bigquery'):
        raise ValueError(f"storage must be parquet, csv, sqlite or bigquery; got {storage_type!r}")

    # create a path/location based on storage type
    location = str(location).replace('\\', '/') # replace slashes

    # resolve dlt destinations
    if storage_type == 'bigquery':
        destination, dataset = dlt.destinations.bigquery(), location
    elif storage_type == 'sqlite':
        destination, dataset = dlt.destinations.sqlalchemy(f"sqlite:///{location}"), Path(location).stem
    else: # else filesystem in specified folder
        parent, _, dataset = location.rstrip('/').rpartition('/')
        destination = dlt.destinations.filesystem(parent or '.')
    return destination, dataset, location

def _last_seen(pipe, table, keys, ts):
    """Gets last stored timestamp (UTC) per key from data source, empty dict if the table doesn't exist yet"""
    if table not in pipe.dataset().tables:
        return {}
    k = ', '.join(keys)
    last = pipe.dataset()(f"SELECT {k}, MAX({ts}) AS last_ts FROM {table} GROUP BY {k}").df().set_index(keys)['last_ts']
    return pd.to_datetime(last, utc=True).to_dict() # store timestamps as UTC; sqlite returns them without tz

# dict of accepted data sources paired with download functions
# on the right are keys (unique identifiers) in each dataset to identify last seen
DATA_SOURCES = {
    'openaq': (get_openaq, ['location_id', 'parameter']),
    'purpleair': (get_purpleair, ['sensor_index']),
    'esdr': (get_esdr, ['feed_id'])
}

STORAGE_TYPES = ('parquet', 'csv', 'sqlite', 'bigquery')

def _check_pipeline_config(config_dict):
    """checks the pipeline parameters in the config are present and valid"""
    for key, options in {'data_source': DATA_SOURCES, 'storage_type': STORAGE_TYPES}.items():
        if config_dict.get(key) not in options:
            raise ValueError(f"{key} must be one of {', '.join(options)}; got {config_dict.get(key)!r}")
    if not isinstance(config_dict.get('location'), str) or not config_dict['location']:
        raise ValueError("location must be a non-empty string with a file path or dataset name")
    
def run_pipeline(config_file:str | Path):
    """downloads new measurements, runs all Air Insights methods and stores measurements and results"""

    # load config dict and parameters
    config_dict = air.load_config(config_file) 
    _check_pipeline_config(config_dict)
    data_source = config_dict['data_source']
    storage_type = config_dict['storage_type']
    location = config_dict['location']

    # based on the inputs, get dlt metadata to run pipeline
    destination, dataset, location = _resolve_storage_destination(storage_type, location)
    file_format = storage_type if storage_type in ('parquet', 'csv') else None

    # determine download function and args for specified dataset
    download_func, keys = DATA_SOURCES[data_source]

    # create pipeline object
    pipe = dlt.pipeline(dataset, destination=destination, dataset_name=dataset)

    # get last stored timestamp per data stream (empty on first run), download newer rows and append
    site, ts = config_dict['site_col'], config_dict['timestamp_col'] 
    seen = _last_seen(pipe, 'measurements', keys, ts)
    pipe.run(download_func(config_dict, seen), table_name='measurements', write_disposition='append', loader_file_format=file_format)

    # read raw measurements from database
    if storage_type == 'bigquery':
        df = pipe.dataset().ibis().table('measurements', database=dataset).to_pyarrow().to_pandas()
    else:
        df = pipe.dataset().table('measurements').df()
    df[ts] = pd.to_datetime(df[ts], utc=True) # make ts timezone-aware before localizing
    df[site] = df.sort_values(ts).groupby(keys[0])[site].transform('last') # one name per site if they change
    
    # run airinsights after formatting data
    # todo: make formatting step an option (read_aqdata_frame)
    results = run_airinsights(_dedupe(_localize_tz(_melt_long(df, config_dict), config_dict), config_dict), config_dict) 

    # only run airtracker when specified in config_dict
    if config_dict.get('airtracker', False):
        # get source area for local event rows, only for site-hours not already stored
        # only go back 365 days for dashboard
        local_events = results['meas_disagg'][results['meas_disagg']['event_ID'].str.startswith('LOC_', na=False)]

        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=365) # don't go back more than a year
        seen_airtracker = _last_seen(pipe, 'source_areas', [site], ts) # last footprint stored per site
        if seen_airtracker:
            cutoff = max(cutoff, max(seen_airtracker.values())) # get latest airtracker value to start from
        new_footprints = local_events[local_events[ts] > cutoff]

        if not new_footprints.empty: # download new footprints
            airtracker = air.source_area(new_footprints, config_dict)[[site, ts, 'wkt_geometry']].drop_duplicates()
            airtracker = airtracker[airtracker['wkt_geometry'].notna()] # exclude failed requests
            pipe.run(dlt.resource(airtracker, name='source_areas', write_disposition='append'), loader_file_format=file_format)

    # replace other results tables
    pipe.run([dlt.resource(out, name=table, write_disposition='replace') for table, out in results.items() if not out.empty],
            loader_file_format=file_format)
    
    return pipe