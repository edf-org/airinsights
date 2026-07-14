import cdsapi
import numpy as np
import pandas as pd
import io
import requests
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def get_era5(input_data:pd.DataFrame,config_dict:dict):
    """Downloads ERA5 wind speed and direction columns for times and locations in AQ measurement dataframe.

    Data Source
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries
    
    Parameters
    ----------
    input_data : pd.DataFrame  
        A pandas DataFrame containing AQ data
    config_dict : dict
        A dictionary containing input parameter names and values. Must contain a cds_api_key field.

    Returns
    -------
    pd.DataFrame
         A pandas DataFrame containing unique site and timestamp columns from the input data with the following columns appended:
            
            **ws**: wind speed (m/s) at closest ERA5 grid cell

            **wd**: wind direction (degrees) at closest ERA5 grid cell
    """
    
    # check config file for CDS API key
    if not config_dict.get('cds_api_key'):
        raise ValueError("'cds_api_key' is required in config file")
    
    # for measurements at each site, get wind data from the closest ERA5 location
    # some sights have moved slightly. take the mode of the measurement location
    df = input_data.copy()
    sites = df.groupby(config_dict['site_col']).agg(
        lat=(config_dict['lat_col'], lambda s: s.mode().iloc[0]),
        lon=(config_dict['lon_col'], lambda s: s.mode().iloc[0]),
        start_date=(config_dict['timestamp_col'], lambda s: s.min().date().isoformat()),
        end_date=(config_dict['timestamp_col'], lambda s: s.max().date().isoformat())).reset_index()
    
    # we know that era5 is 0.25 deg, so drop duplicate sites to speed up download
    sites["grid_lat"] = ((sites["lat"] / 0.25).round() * 0.25).round(2)
    sites["grid_lon"] = ((sites["lon"] / 0.25).round() * 0.25).round(2)
    sites_gridded = sites.groupby(["grid_lat", "grid_lon"]).agg(
        start_date=("start_date", "min"),
        end_date=("end_date", "max")).reset_index()
    
    def parallel_download(row,config_dict): # changing it to be like get_airtracker ,start_date,lat,lon,site
        
        # query cds API for closest ERA5 point to specified lat/lon
        result = cdsapi.Client(url = 'https://cds.climate.copernicus.eu/api',key=config_dict['cds_api_key']).retrieve(
            "reanalysis-era5-single-levels-timeseries",
            {
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "location": {"latitude": row.grid_lat, "longitude": row.grid_lon},
                "date": [f"{row.start_date}/{row.end_date}"], # start date thru today
                "data_format": "csv",
            },
        )

        # download and calculate wind speed and direction from u and v components
        temp = pd.read_csv(io.BytesIO(requests.get(result.location).content), compression="zip")
        temp["ws"] = np.sqrt(temp["u10"] ** 2 + temp["v10"] ** 2)
        temp["wd"] = (270.0 - np.degrees(np.arctan2(temp["v10"], temp["u10"]))) % 360.0
        temp[config_dict['timestamp_col']] = pd.to_datetime(temp["valid_time"], utc=True)
        
        # to join back to sites
        temp["grid_lat"] = row.grid_lat
        temp["grid_lon"] = row.grid_lon
        
        print(f"finished grid point ({row.grid_lat}, {row.grid_lon})")
        
        # keep site and timestamp to join back
        return temp[["grid_lat", "grid_lon", config_dict['timestamp_col'], "ws", "wd"]]
            
    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(parallel_download, row, config) for row in sites_gridded.itertuples()]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"Failed with error: {e}")
    
    out = pd.concat(results, ignore_index=True)
    
    # join back: every site gets the wind timeseries from its grid point
    out = sites[[config_dict['site_col'], "grid_lat", "grid_lon"]].merge(
        out, on=["grid_lat", "grid_lon"], how="left")
    
    return out[[config_dict['site_col'], config_dict['timestamp_col'], "ws", "wd"]]