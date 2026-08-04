from rasterio.features import shapes
import geopandas as gpd
import pandas as pd
import concurrent.futures
from rasterio.io import MemoryFile
import requests
import rioxarray
from shapely.geometry import shape
from shapely.ops import unary_union

def batch_airtracker(input_data:pd.DataFrame,config_dict:dict,max_workers: int = 16):
    """ Get airtracker in parallel, one per row of input_data with times and coordinates
    Returns geodataframe with geometry of airtracker footprint for each measurement (row)
    """
    
    # for each row in pollution event df, get a footprint
    def run_once(row):
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
        futures = [executor.submit(run_once, row) for row in input_data.itertuples()]
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