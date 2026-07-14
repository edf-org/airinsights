import rasterio
import urllib.request
from rasterio.features import shapes
import geopandas as gp
import pandas as pd
import numpy as np
import concurrent.futures
import tempfile
import os

def get_airtracker(input_data:pd.DataFrame,config_dict:dict):
    
    # for each local event timestamp, lat, and lon, get a footprint

    df = input_data.copy()
        
    def parallel_download(row,config_dict):
        time = getattr(row,config_dict['timestamp_col']).strftime('%Y-%m-%dT%H:%M')
        lon = str(getattr(row,config_dict['lon_col']))
        lat = str(getattr(row,config_dict['lat_col']))
        url = 'https://api.airtracker.createlab.org/get_footprint?view=' + lat + ',' + lon + ',' + '12&time=' + time + '&showContributionLikelihoods=true&isRealTime=true&format=geotiff'            
        with tempfile.NamedTemporaryFile(suffix='.geotiff', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            urllib.request.urlretrieve(url, tmp_path) # write temp file of geotiff
            with rasterio.Env():
                with rasterio.open(tmp_path) as src:
                    image = src.read(1)
                    background = np.bincount(image.flatten()).argmax() # raster background is single value
                    binary_image = (image != background).astype(np.uint8)  # 1 or 0 for if it is in the region
                    geoms = [{'properties': {'raster_val': v}, 'geometry': s} for _, (s, v) in enumerate(shapes(binary_image, mask=binary_image, transform=src.transform))] 
            gdf = gp.GeoDataFrame.from_features(geoms,crs='EPSG:3857') #Convert to Geopandas dataframe
            gdf = gdf.to_crs("EPSG:4326")
            gdf['wkt_geometry'] = gdf['geometry'].to_wkt()
            gdf['event_ID'] = getattr(row,'event_ID')
            gdf[config_dict['site_col']] = getattr(row,config_dict['site_col'])
            gdf[config_dict['lat_col']] = getattr(row,config_dict['lat_col'])
            gdf[config_dict['lon_col']] = getattr(row,config_dict['lon_col'])
            gdf[config_dict['timestamp_col']] = getattr(row,config_dict['timestamp_col'])
            gdf = gdf.drop(columns=['geometry'])
            print(f"finished {time}")
            return(gdf)
        finally:
            os.unlink(tmp_path)
    
    results = [] 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(parallel_download, row, config_dict) for row in df.itertuples()]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"Failed with error: {e}")
    
    non_empty = [df for df in results if not df.empty]

    if non_empty:
        out = pd.concat(non_empty, ignore_index=True)
    else:
        out = pd.DataFrame(columns=['raster_val', 
                                    'wkt_geometry', 
                                    'event_ID',
                                    config_dict['site_col'],
                                    config_dict['lat_col'],
                                    config_dict['lon_col'],
                                    config_dict['timestamp_col']])
    return out