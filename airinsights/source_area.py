from rasterio.features import shapes
import geopandas as gpd
import pandas as pd
import concurrent.futures
from rasterio.io import MemoryFile
import requests
import rioxarray
from shapely.geometry import shape
from shapely.ops import unary_union

def get_airtracker(input_data:pd.DataFrame,config_dict:dict,max_workers: int = 16):
    """Get airtracker footprint for each measurement in a dataset, in parallel

    Calls airtracker_footprint once per row of input_data using each row's timestamp and coordinates.
    Typically run on the disaggregated output of classify_pollution_events to identify likely upwind source areas for 
    local pollution events.
    
    Parameters
    ----------
    input_data : pd.DataFrame
        A pandas DataFrame with one row per measurement, containing timestamp, latitude, longitude, site, and event_ID columns.
    config_dict : dict
        A dictionary containing input parameter names and values. See 'Other Parameters' for a list.
    max_workers : int, default 16
        Number of concurrent threads used to request footprints from the AirTracker API.

    Returns
    -------
    gpd.GeoDataFrame or None
        A GeoDataFrame with one row per successfully retrieved footprint, containing the footprint geometry (as both
        a shapely geometry and a WKT string), timestamp, longitude, latitude, event_ID, and site columns.

    Other Parameters
    ----------------
    timestamp_col : str
        Name of the column containing date and time
    lat_col : str
        Name of the latitude column
    lon_col : str
        Name of the longitude column
    site_col : str or int
        Name of the column containing unique identifiers for the air sensors
    """

    # for each row in pollution event df, get a footprint
    def _run_once(row):
        gdf = airtracker_footprint(
            time=getattr(row, config_dict['timestamp_col']),
            lon=getattr(row, config_dict['lon_col']),
            lat=getattr(row, config_dict['lat_col']),
            output="vector"
        )
        gdf['event_ID'] = getattr(row, 'event_ID')
        gdf[config_dict['site_col']] = getattr(row, config_dict['site_col'])
        return gdf

    results = [] 
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_once, row) for row in input_data.itertuples()]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"Failed with error: {e}")
    
    non_empty = [df for df in results if not df.empty]
    if non_empty:
        return pd.concat(non_empty, ignore_index=True)

def airtracker_footprint(time:pd.Timestamp,lon:float,lat:float,output = "vector"):
    """Get an airtracker footprint for a single time and location. Option to return as raster or vector."""
    
    # convert to format airtracker expects - it is in local time, so this converts to string while maintaining local
    time = pd.Timestamp(time).strftime("%Y-%m-%dT%H:%M") 
    
    if output not in ("vector", "raster"):
        raise ValueError(f"output must be 'vector' or 'raster', got {output!r}")
    
    url = (f'https://api.airtracker.createlab.org/get_footprint?view={lat},{lon},12&time={time}' 
           '&showContributionLikelihoods=true&isRealTime=true&format=geotiff')
    
    content = requests.get(url)
    content.raise_for_status()
    
    with MemoryFile(content.content) as memfile, memfile.open() as src:
        raster = rioxarray.open_rasterio(src).isel(band=3).load() # select band that shows whole footprint (not relative strength)

    if output == "raster":
        return raster
    
    elif output == "vector":
        mask = raster.values > 0
        geoms = [shape(s) for s, _ in shapes(mask.astype("uint8"),mask=mask, transform=raster.rio.transform())]
        merged = unary_union(geoms) if geoms else None
        gdf = gpd.GeoDataFrame({"time": [time], "lon": [lon], "lat": [lat]},
                               geometry=[merged], crs=raster.rio.crs).to_crs("EPSG:4326")
        gdf['wkt_geometry'] = gdf['geometry'].to_wkt()
        #gdf = gdf.drop(columns=['geometry']) if we want to keep WKT only
        return gdf