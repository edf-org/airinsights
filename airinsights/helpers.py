import yaml
from pathlib import Path
import pandas as pd
import importlib.resources

# TODO - should we take out default config? the configs are location, timezone specific so likely won't be applicable to a random user.

STANDARD_POLLUTANTS = {
    "BC",
    "PM1",
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "O3",
    "CO",
    "SO2"
}

def build_config(
    timestamp_col: str,
    timestamp_tz: str,
    local_tz : str,
    site_col: str,
    value_col: str,
    config_file_path: str,
    lat_col: str,
    lon_col: str,
    wide_format: bool = True,
    timestamp_format: str = "%m/%d/%Y %H:%M",
    pollutants: dict[str,str] = {},
    pollutant_col: str | None = None,
    file_delimiter: str | None = None,
    output_file_path: str | None = None,
    confidence_col: str | None = None,
    confidence_threshold: int | float | None=None

    
) -> dict:
    """
    Builds a new yaml config file and returns the file path to load and analyze data with AirInsights.

    Parameters
    ----------
    timestamp_col : str 
        Name of the column containing date and time
    timestamp_tz : str
        Timezone of the timestamp column. For example: "America/Los_Angeles". For more info, see: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
    local_tz : str
        Timezone to convert the timestamp column to. For example: "America/Los_Angeles". For more info, see: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
    site_col : str
        Name of the column containing unique identifiers for the air sensors
    value_col : str 
        Name of the column containing the values for the pollutant of interest 
    lat_col : str
        Name of the latitude column
    lon_col : str
        Name of the longitude column
    wide_format : bool, optional
        Optional. If True, the input data is in wide format and will be pivoted to long format using the pollutant columns specified in the pollutants dictionary. If False, the input data is already in long format and will not be pivoted.
    timestamp_format: str
        Format of the timestamp column usign python datetime syntax. For example: 2026-01-01 09:27:11 -> %Y-%m-%d %H:%M:%S. For more info, see: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    pollutants : dict
        Required. Dictionary specifying the pollutant(s) to analyze. The keys must
        be standard Airinsights pollutant names (e.g., "PM2.5", "NO2", "O3"),
        and the values are the corresponding pollutant column names in the input
        data. For example: {"PM2.5": "pm25_concentration", "NO2": "no2_ppb"}.
    pollutant_col : str | None, optional
        Optional. Name of the column containing the pollutant names
    config_file_path : str
        If path ending .yaml provided to write config file.
    file_delimiter : str, optional
        Delimiter of AQ file to load.
    output_file_path : str, optional
        Path to write outputs of AirInsights functions.
    confidence_col : str, optional
        Optional. Column in AQ file containing measurement confidence values
    confidence_threshold : int or float, optional
        Optional. Minimum confidence value measurements must meet to be included in the analysis

    Returns
    -------
        config_dict: configuration dictionary
    """
        
    file_ext = Path(config_file_path).suffix.lower()
    if not file_ext == ".yaml":
        raise ValueError("config file path must end in .yaml")

    yaml_pollutants = None
    if pollutants is not None:
        yaml_pollutants = {
            pollutant: {
                "name": name,
            }
            for pollutant, name in pollutants.items()
        }

    config_dict = {
        "timestamp_col": timestamp_col,
        "timestamp_tz": timestamp_tz,
        "local_tz": local_tz,
        "site_col": site_col,
        "value_col": value_col,
        "timestamp_format": timestamp_format,
        "lat_col" : lat_col,
        "lon_col" : lon_col,
        "wide_format": wide_format,
        "file_delimiter": file_delimiter,
        "output_file_path": output_file_path,
        "confidence_col": confidence_col,
        "confidence_threshold": confidence_threshold,
        "pollutants": yaml_pollutants,
        "pollutant_col": pollutant_col,
    }
    
    config_path = Path(config_file_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote config to {config_file_path}")

    return config_file_path

def load_config(
    config : str | Path
) -> dict:
    """Loads a YAML configuration file and checks for required parameters

    Parameters
    ----------
    config : str or Path
        path to a YAML configuration file
    
    Returns
    -------
    dict
        A dictionary containing input parameter names and values

    """
    config_path = Path(config)
    # --- Read in YAML config file ---
    with open(config_path, 'r') as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.scanner.ScannerError:
            raise
        except FileNotFoundError:
            print(f"Error: The file {config} was not found.")
        except yaml.YAMLError as yaml_error:
            print(f"Error parsing YAML file: {yaml_error}")
    # --- Check for required parameters --- 
    try:
        config_dict['timestamp_col']
        config_dict['timestamp_format']
        config_dict['timestamp_tz']
        config_dict['local_tz']
        config_dict['wide_format'] 
        config_dict['pollutants']
        config_dict['site_col']
        config_dict['lat_col']
        config_dict['lon_col']
    except KeyError as missing_key:
        print(f"Error: {missing_key} is missing. Check the configuration file.")
        raise

     # Check format-specific parameters
    if not config_dict["wide_format"]:
        pollutant_col = config_dict.get("pollutant_col")
        value_col = config_dict.get("value_col")

        if not pollutant_col or not value_col:
            missing = "pollutant_col" if not pollutant_col else "value_col"

            raise ValueError(
                f"Data is in long format, but no {missing} is provided. "
                "Check the configuration file."
            )

    return config_dict

def read_aqdata_file(
    input_file : str | Path,
    config : str | Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Reads in AQ data file to a pandas DataFrame, then formats the DataFrame using inputs from a YAML configuration file
    Supported AQ data file formats are csv, json, and excel files (xsl, xlsx, xlsm)

    Parameters
    ----------
    input_file : str or Path    
        path to the AQ data file
    config : str or Path, default config/100x100_config.yaml
        path to a YAML configuration file

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the input AQ data
    dict
        A dictionary containing input parameter names and values
    """
    # --- Convert input_file string to Path type if needed
    data_path = Path(input_file)
    # --- Load a default configuration if not specified in the function call
    if config is None:
        print('No configuration file specified. Using the default.')
        with importlib.resources.path("airinsights", 'config/100x100_config.yaml') as default_config:
            config_path = default_config
    else:
        config_path = Path(config)

    # --- Load a configuration file to ensure correct formatting on read
    config_dict = load_config(config_path)

    # --- Take the file extension to determine which pandas function to use    
    suffixes = [s.lower() for s in Path(data_path).suffixes]
    file_ext = "".join(suffixes)
    
    if file_ext in ('.csv', '.csv.gz'):
        df = pd.read_csv(input_file) 
    elif file_ext in ('.xls', '.xlsx', '.xlsm'):
        df = pd.read_excel(input_file)
    elif file_ext == '.json':
        df = pd.read_json(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported file formats are csv, json, and excel files (xsl, xlsx, xlsm)")
    
    # --- Format date column using config. This will throw error if it fails
    df[config_dict['timestamp_col']] = pd.to_datetime(df[config_dict['timestamp_col']],format=config_dict['timestamp_format'])
    
    df = _melt_long(df,config_dict)
    df = _localize_tz(df,config_dict)
    df = _dedupe(df,config_dict)
    
    return df, config_dict

def read_aqdata_bq(
    input_table:str,
    config : str | Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Reads an AQ data table from BigQuery to a pandas DataFrame, then formats the DataFrame using inputs from a YAML configuration file"""
    
    from google.cloud import bigquery # only load when func runs

    # --- Load a default configuration if not specified in the function call
    if config is None:
        print('No configuration file specified. Using the default.')
        with importlib.resources.path("airinsights", 'config/100x100_config.yaml') as default_config:
            config_path = default_config
    else:
        config_path = Path(config)

    config_dict = load_config(config_path)

    client = bigquery.Client()
    df = client.list_rows(input_table).to_dataframe()

    # --- Shared with read_aqdata_file
    df = _melt_long(df, config_dict)
    df = _localize_tz(df, config_dict)
    df = _dedupe(df, config_dict)
    
    return df, config_dict

def _melt_long(df:pd.DataFrame,config_dict:dict) -> pd.DataFrame:
    """If data is wide format, pivot to long using specified columns"""
    if not config_dict['wide_format']:
        return df
    
    print("Melting data to long format")
    
    # get pollutant and value_col names for pivot, using default if not supplied in config
    pollutant_col = config_dict.get('pollutant_col')
    if not pollutant_col:
        print("No pollutant_col in config; using default name 'pollutant'")
        config_dict['pollutant_col'] = 'pollutant'

    value_col = config_dict.get('value_col')
    if not value_col:
        print("No value_col in config; using default name 'value'")
        config_dict['value_col'] = 'value'

    value_vars = [v['name'] for v in config_dict['pollutants'].values()]
        
    return df.melt(id_vars=[c for c in df.columns if c not in value_vars],
                    value_vars = value_vars,
                    var_name=config_dict['pollutant_col'],
                    value_name=config_dict['value_col'])

def _localize_tz(df:pd.DataFrame,config_dict:dict) -> pd.DataFrame:
    """ If tz specified in config, localize the column"""
    # TODO this could be made automatic based on lat/lon of data

    ts_col = df[config_dict['timestamp_col']]

    if not isinstance(ts_col.dtype, pd.DatetimeTZDtype): # if there is no timezone in pandas, assign the correct one from config
        ts_col = ts_col.dt.tz_localize(config_dict['timestamp_tz'])
    
    if str(ts_col.dt.tz) == config_dict['local_tz']:
        print(f"Timestamp already in local timezone: {config_dict['local_tz']}")
    else:
        print(f"Converting timestamp to local timezone: {config_dict['local_tz']}")
        ts_col = ts_col.dt.tz_convert(config_dict['local_tz']) # then convert to local_tz
        
    df[config_dict['timestamp_col']] = ts_col

    return df

def _dedupe(df:pd.DataFrame,config_dict:dict) -> pd.DataFrame:
    """Remove exact duplicate measurements (same site, timestamp, pollutant, and value)"""
    original_length = len(df)
    df = df.drop_duplicates(subset = [config_dict['site_col'],
                                      config_dict['timestamp_col'],
                                      config_dict['pollutant_col'],
                                      config_dict['value_col']])
    removed = original_length - len(df)
    if removed:
        print(f"Removed {removed} exact duplicate rows")
    return df

def _infer_temporal_freq(t):
    """Infer frequency of measurements"""
    diffs = t.sort_values().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]  # drop zero diffs (duplicate timestamps)

    return pd.Timedelta(pd.tseries.frequencies.to_offset(diffs.mode().iloc[0]))

def _make_hourly(df,config_dict):
    """Check time resolution of measurements and average to hourly or throw error"""
    df = df.copy()
    freqs = df.groupby([config_dict['site_col'],config_dict['pollutant_col']])[config_dict['timestamp_col']].apply(_infer_temporal_freq)

    # exclude data that is less frequent than hourly
    too_infrequent = freqs[freqs > pd.Timedelta(hours=1)]
    if not too_infrequent.empty:
        df = df[~df.set_index([config_dict['site_col'],config_dict['pollutant_col']]).index.isin(too_infrequent.index)]
        print(f"Excluded data from {len(too_infrequent)} loc/param combinations with time resolution less frequent than hourly")

    # error if all data was less frequent than hourly
    if df.empty:
        raise ValueError("All data were excluded as too infrequent (interval > 1h) for this method")

    # average data that is more frequent than hourly
    sub_hourly = freqs[freqs < pd.Timedelta(hours=1)]
    if not sub_hourly.empty:
        value_col = config_dict['value_col']
        group_cols = [c for c in df.columns if c not in [value_col, config_dict['timestamp_col']]]
        df = (
            df.set_index(config_dict['timestamp_col'])
            .groupby(group_cols)
            .resample('h')[value_col]
            .mean()
            .dropna()
            .reset_index()
            )
        print(f"Resampled {len(sub_hourly)} loc/param combinations to hourly mean")
    
    return(df)