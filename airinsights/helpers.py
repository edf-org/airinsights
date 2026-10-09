# Copyright 2026 Environmental Defense Fund, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
from pathlib import Path
import pandas as pd
import warnings

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
    local_tz : str,
    timestamp_col: str = 'datetime',
    timestamp_format: str = '%Y-%m-%d %H:%M:%S+00:00',
    timestamp_tz: str = 'UTC',
    site_col: str = 'site_name',
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    pollutant_map: dict[str,str] = {"BC": "bc",
                                    "PM1": "pm1",
                                    "PM2.5": "pm25",
                                    "PM10": "pm10",
                                    "NO": "no",
                                    "NO2": "no2",
                                    "NOx": "nox",
                                    "O3": "o3",
                                    "CO": "co",
                                    "SO2": "so2"},
    value_col: str = 'value',
    config_file: str = 'config/example_config.yaml'
) -> str:
    """
    Builds a new yaml config file and writes to specified file path to load and analyze data with AirInsights.
    
    See examples/airinsights_setup.ipynb on GitHub for a full working example: 
    https://github.com/edf-org/airinsights/blob/main/examples/airinsights_setup.ipynb

    Parameters
    ----------
    local_tz : str
        Local timezone to convert the timestamp column to. For example: "America/Los_Angeles". 
        For more info, see: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
    timestamp_col : str, default 'datetime' 
        Name of the column containing date and time
    timestamp_format: str, default '%Y-%m-%d %H:%M:%S+00:00'
        Format of the timestamp column usign python datetime syntax. 
        For example: 2026-01-01 09:27:11 -> %Y-%m-%d %H:%M:%S. 
        For more info, see: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    timestamp_tz : str, default 'UTC'
        Timezone of the timestamp column. For example: "UTC". 
        For more info, see: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
    site_col : str, default 'site_name'
        Name of the column containing unique identifiers for the air sensors
    lat_col : str, default 'lat'
        Name of the latitude column
    lon_col : str, default 'lon'
        Name of the longitude column
    pollutant_map :
        Dictionary specifying the pollutant(s) to analyze. The keys must
        be standard Airinsights pollutant names (e.g., "PM2.5", "NO2", "O3"),
        and the values are the corresponding pollutant column names in the input
        data. For example: {"PM2.5": "pm25", "NO2": "no2"}.
    value_col : str, default 'value'
        Applies only when input data is in long format. Name of the column containing the measurement values. 
    config_file : str, default 'config/example_config.yaml'
        File path ending in .yaml to write config file.
    """
    config_path = Path(config_file)    
    file_ext = config_path.suffix.lower()
    if file_ext != ".yaml":
        raise ValueError("config file path must end in .yaml")

    if not pollutant_map:
        raise ValueError("pollutant_map is required but missing. See function documentation for details.") 

    config_dict = {
        "local_tz": local_tz,
        "timestamp_col": timestamp_col,
        "timestamp_format": timestamp_format,
        "timestamp_tz": timestamp_tz,
        "site_col": site_col,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "pollutant_map": dict(pollutant_map),
        "value_col": value_col,
    }
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
    print(f"Wrote config to {config_file}")

    return None

def load_config(
    config_file : str | Path
) -> dict:
    """Loads a YAML configuration file and checks for required parameters.
    
    See examples/airinsights_setup.ipynb on GitHub for a full working example: 
    https://github.com/edf-org/airinsights/blob/main/examples/airinsights_setup.ipynb


    Parameters
    ----------
    config_file_path : str or Path
        path to a YAML configuration file
    
    Returns
    -------
    dict
        A dictionary containing input parameter names and values

    """
    config_path = Path(config_file)

    # --- Read in YAML config file ---
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # --- Check for required parameters --- 
    required = {"local_tz": str, "timestamp_col": str, "timestamp_format": str, "timestamp_tz": str,
                "site_col": str, "lat_col": str, "lon_col": str, "pollutant_map": dict}
    
    # first, are any missing entirely?
    missing = [col for col in required if col not in config_dict]
    if missing:
        raise ValueError(f"Config is missing the required parameter(s): {', '.join(missing)}")

    # second, are any empty or the wrong type?
    invalid = [col for col, data_type in required.items() 
               if not isinstance(config_dict[col], data_type) # is data type wrong?
               or not config_dict[col]] # or is it empty?
    if invalid:
        raise ValueError(f"Config has empty or invalid parameter(s): {', '.join(invalid)}")

    # third, is pollutant_map correct? does it have key value pairs with nothing empty?
    pollutant_map_errors = [standard_pollutant for standard_pollutant, pollutant_name in config_dict["pollutant_map"].items()
                            if not standard_pollutant or not pollutant_name]
    if pollutant_map_errors:
        raise ValueError("pollutant_map has invalid entries, check config. Each pollutant requires a standard name and name from input data, e.g. {'PM2.5': 'pm25'}.")

    # fourth, do pollutant_map keys match standard pollutant names?
    non_standard = set(config_dict["pollutant_map"]) - STANDARD_POLLUTANTS # any non-standard pollutants?
    if non_standard:
        warnings.warn(
            f"pollutant_map has non-standard pollutant name(s): {', '.join(sorted(map(str, non_standard)))}. "
            f"Standard names are: {', '.join(sorted(STANDARD_POLLUTANTS))}"
        )

    return config_dict

def read_aqdata_file(
    input_file : str | Path,
    config_file : str | Path
) -> tuple[pd.DataFrame, dict]:
    """Reads in AQ data file to a pandas DataFrame, then formats the DataFrame using inputs from a YAML configuration file.
    Supported AQ data file formats are csv, json, and excel files (xls, xlsx, xlsm).

    See examples/airinsights_setup.ipynb on GitHub for a full working example: 
    https://github.com/edf-org/airinsights/blob/main/examples/airinsights_setup.ipynb

    Parameters
    ----------
    input_file : str or Path    
        path to the AQ data file
    config_file : str or Path
        path to a YAML configuration file

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the input AQ data
    dict
        A dictionary containing input parameter names and values
    """
    
    # --- Load a configuration file to ensure correct formatting on read
    config_dict = load_config(Path(config_file))

    # --- Take the file extension to determine which pandas function to use    
    suffixes = [s.lower() for s in Path(input_file).suffixes]
    file_ext = "".join(suffixes)
    
    if file_ext in ('.csv', '.csv.gz'):
        df = pd.read_csv(input_file) 
    elif file_ext in ('.xls', '.xlsx', '.xlsm'):
        df = pd.read_excel(input_file)
    elif file_ext == '.json':
        df = pd.read_json(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported file formats are csv, json, and excel files (xls, xlsx, xlsm)")
    
    # --- Format date column using config. This will throw error if it fails
    df[config_dict['timestamp_col']] = pd.to_datetime(df[config_dict['timestamp_col']],format=config_dict['timestamp_format'])

    df = _melt_long(df,config_dict)
    df = _localize_tz(df,config_dict)
    df = _dedupe(df,config_dict)
    
    return df, config_dict

def read_aqdata_bq(
    input_table:str,
    config_file : str | Path 
) -> tuple[pd.DataFrame, dict]:
    """Reads an AQ data table from BigQuery to a pandas DataFrame, then formats the DataFrame using inputs from a YAML configuration file"""
    
    from google.cloud import bigquery # only load when func runs

    config_dict = load_config(Path(config_file))

    client = bigquery.Client()
    df = client.list_rows(input_table).to_dataframe()

    # --- Shared with read_aqdata_file
    df = _melt_long(df, config_dict)
    df = _localize_tz(df, config_dict)
    df = _dedupe(df, config_dict)
    
    return df, config_dict

def _infer_format(df:pd.DataFrame,config_dict:dict):
    """Infer whether AQ dataframe is in long or wide format using the config pollutant names"""
    input_pollutants = set(config_dict['pollutant_map'].values()) # get input names from pollutant map
    matching_cols = input_pollutants & set(df.columns) # compare with column names

    if matching_cols: # if any matches, data is wide format
        is_wide = True 
        pollutant_col = 'pollutant' # not in wide data, assign default name
        value_col = 'value' # not in wide data, assign default name
        missing = input_pollutants - matching_cols  # any pollutants not in column names?
        if missing:
            warnings.warn(f"Some user specified pollutant columns were not found: {', '.join(sorted(missing))}. Check pollutant_map in the config file.")
        
    else: # else is long format
        is_wide = False
        # look for 'pollutant' column containing pollutant names
        str_cols = df.select_dtypes(include=["string","object"]).columns # select string cols
        pollutant_cols = [col for col in str_cols if df[col].isin(input_pollutants).any()]  # select columns with input pollutant names in contents

        # error if no string columns containing pollutant names
        if not pollutant_cols:
            raise ValueError(
                "No column contains any of the user specified pollutant names. Check pollutant_map in the config file."
            )

        # if more than one column was found with pollutant names in contents, warn (use the first)
        pollutant_col = pollutant_cols[0] # select one that matches the names
        if len(pollutant_cols) > 1:
            warnings.warn(f"Found more than one column containing pollutant names, using '{pollutant_col}'")

        # warn about pollutants in the map that aren't in the data
        missing = input_pollutants - set(df[pollutant_col])
        if missing:
            warnings.warn(f"Pollutants not found in the data, skipping: {', '.join(sorted(missing))}")
            
        # long format needs to know which column holds the values
        value_col = config_dict.get('value_col') or 'value'  # default to 'value' if not in config
        if value_col not in df.columns:
            raise ValueError(f"No '{value_col}' column found in the data. Set value_col in the config to the column with measurement values.")

    return is_wide, pollutant_col, value_col

def _melt_long(df:pd.DataFrame,config_dict:dict) -> pd.DataFrame:
    """If data is wide format, pivot to long using specified columns"""

    # infer the data format based on the pollutant_map in config
    is_wide, pollutant_col, value_col = _infer_format(df, config_dict)

    # if data is wide format, melt to long
    if is_wide:
        value_vars = [col for col in config_dict['pollutant_map'].values() if col in df.columns] # get column headers from pollutant map if in data
        df = df.melt(id_vars=[c for c in df.columns if c not in value_vars],
                    value_vars = value_vars,
                    var_name=pollutant_col,
                    value_name=value_col)
        
    # otherwise it's long format; keep only the pollutants in the map
    else:  
        df = df[df[pollutant_col].isin(config_dict['pollutant_map'].values())]

    # update pollutant and value column names of long format data in config_dict
    config_dict['value_col'] = value_col
    config_dict['pollutant_col'] = pollutant_col

    return df

def _localize_tz(df:pd.DataFrame,config_dict:dict) -> pd.DataFrame:
    """ Localize the column to tz specified in config """
    # TODO this could be made automatic based on lat/lon of data

    ts_col = df[config_dict['timestamp_col']]

    if not isinstance(ts_col.dtype, pd.DatetimeTZDtype): # if there is no timezone in pandas, assign the correct one from config
        ts_col = ts_col.dt.tz_localize(config_dict['timestamp_tz'], ambiguous='NaT', nonexistent='NaT')
        # drop times that are repeated or skipped at DST changes
        if ts_col.isna().any(): 
            warnings.warn(f"Dropped {ts_col.isna().sum()} rows with missing or ambiguous local times (e.g. DST transitions)")
            df, ts_col = df[ts_col.notna()], ts_col[ts_col.notna()]

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
    """Infer frequency of measurements  
    
    Parameters  
    ----------  
    t : pd.Series 
        Timestamp column for a single site and pollutant  

    Returns  
    -------  
    pd.Timedelta  
        Duration between measurements  
    """  
    diffs = t.sort_values().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]  # drop zero diffs (duplicate timestamps)
    return diffs.mode().iloc[0] if not diffs.empty else pd.NaT

# TODO - make this just for averaging now that we have _validate_hourly, and add thresholds
# implement in trends function to reduce duplication
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

def _validate_hourly(df,config_dict):
    """Check time resolution of measurements and exclude data that is not exactly hourly; error if no hourly data remains"""
    df = df.copy()
    freqs = df.groupby([config_dict['site_col'],config_dict['pollutant_col']])[config_dict['timestamp_col']].apply(_infer_temporal_freq)

    # exclude site/pollutant combinations that are not exactly hourly
    non_hourly = freqs[freqs != pd.Timedelta(hours=1)]
    if not non_hourly.empty:
        df = df[~df.set_index([config_dict['site_col'],config_dict['pollutant_col']]).index.isin(non_hourly.index)]
        print(f"Excluded data from {len(non_hourly)} site/pollutant combinations with resolution other than hourly:\n{non_hourly}")

    # error if all data was excluded
    if df.empty:
        raise ValueError("All data were excluded as non-hourly; this function requires hourly data.")

    return(df)