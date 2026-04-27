import yaml
from pathlib import Path
import pandas as pd
import importlib.resources

def build_config(
    timestamp_col: str,
    site_col: str,
    value_col: str,
    config_file_path: str,
    lat_col: str,
    lon_col: str,
    datetime_format: str = "%m/%d/%Y %H:%M",
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
    site_col : str
        Name of the column containing unique identifiers for the air sensors
    value_col : str 
        Name of the column containing the values for the pollutant of interest 
    lat_col : str
        Name of the latitude column
    lon_col : str
        Name of the longitude column
    datetime_format: str
        Format of the timestamp column usign python datetime syntax. For example: 2026-01-01 09:27:11 -> %Y-%m-%d %H:%M:%S. For more info, see: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
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
    
    config_dict = {
        "timestamp_col": timestamp_col,
        "site_col": site_col,
        "value_col": value_col,
        "datetime_format": datetime_format,
        "lat_col" : lat_col,
        "lon_col" : lon_col,
        "file_delimiter": file_delimiter,
        "output_file_path": output_file_path,
        "confidence_col": confidence_col,
        "confidence_threshold": confidence_threshold,
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
        config_dict['site_col']
        config_dict['value_col'] # could be a string or a list, if it's a list then run through all the columns
        config_dict['datetime_format']
        config_dict['lat_col']
        config_dict['lon_col']
    except KeyError as missing_key:
        print(f"Error: {missing_key} is missing. Check the configuration file.")

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
        print(f'No configuration file specified. Using the default.')
        with importlib.resources.path("airinsights", 'config/100x100_config.yaml') as default_config:
            config_path = default_config
    else:
        config_path = Path(config)

    # --- Load a configuration file to ensure correct formatting on read
    config_dict = load_config(config_path)

    # --- Take the file extension to determine which pandas function to use
    file_ext = Path(data_path).suffix.lower()
    if file_ext == '.csv':
        df = pd.read_csv(input_file, parse_dates=[config_dict['timestamp_col']], date_format=config_dict['datetime_format'],
                        dtype={config_dict['value_col']: float}) 
    elif file_ext in ('.xls', '.xlsx', '.xlsm'):
        df = pd.read_excel(input_file, parse_dates=[config_dict['timestamp_col']], date_format=config_dict['datetime_format'],
                        dtype={config_dict['value_col']: float})
    elif file_ext == '.json':
        df = pd.read_json(input_file, dtype={config_dict['value_col']: float})
        df[config_dict['timestamp_col']] = pd.to_datetime(df[config_dict['timestamp_col']], format=config_dict['datetime_format'])
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported file formats are csv, json, and excel files (xsl, xlsx, xlsm)")
    
    return df, config_dict