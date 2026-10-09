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

from rasterio.features import shapes
import geopandas as gpd
import pandas as pd
import concurrent.futures
from rasterio.io import MemoryFile
import requests
import rioxarray
from shapely.geometry import shape
from shapely.ops import unary_union
import warnings

def source_area(input_data: pd.DataFrame,
                config_dict: dict,
                max_workers: int = 16
                ):
    """Get airtracker footprint for each measurement in a dataset, in parallel

    Calls _airtracker_footprint once per row of input_data using each row's timestamp and coordinates.
    Typically run on the disaggregated output of pollution_events() to identify likely upwind source areas for 
    local pollution events. Air Tracker is currently available only in certain locations; for more information about Air Tracker
    methods and locations please visit: https://www.edf.org/air-tracker-mapping-local-air-pollution. 
    
    Parameters
    ----------
    input_data : pd.DataFrame
        A pandas DataFrame with one row per measurement, containing timestamp, latitude, 
        longitude, site, pollutant, and event_ID columns.
    config_dict : dict
        A dictionary containing input parameter names and values. See 'Other Parameters' for a list.
    max_workers : int, default 16
        Number of concurrent threads used to request footprints from the AirTracker API.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with one row per successfully retrieved footprint, containing the footprint geometry (as both
        a shapely geometry and a WKT string), timestamp, site, longitude, latitude, pollutant, and event_ID columns.

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
    # columns names from df to pass to _airtracker_footprint function (one row at a time)
    time, lat, lon = config_dict['timestamp_col'], config_dict['lat_col'], config_dict['lon_col']
    unique = input_data[[time, lat, lon]].drop_duplicates() # only run on unique airtracker footprints

    # initiate parallel run of _airtracker_footprint by unique time and location
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_airtracker_footprint,unique[time],unique[lat],unique[lon]) # args passed in order
        geoms = [r.geometry.iloc[0] if r is not None else None for r in results] # get geometry for each run 

    # add geoms to input_data, including wkt string
    gdf = gpd.GeoDataFrame(input_data[[time, lat, lon, config_dict['site_col'],config_dict['pollutant_col'], 'event_ID']] # select return cols
                           .merge(unique.assign(geometry=geoms), on=[time, lat, lon], how='left'),
                           geometry='geometry', crs="EPSG:4326")
    gdf['wkt_geometry'] = gdf['geometry'].to_wkt()

    return gdf

def _airtracker_footprint(time: pd.Timestamp,
                         lat: float,
                         lon: float,
                         output: str = "vector"):
    """Get an airtracker footprint for a single time and location. Option to return as raster or vector."""
    
    # convert to format airtracker expects - it is in local time, so this converts to string while maintaining local
    time_str = pd.Timestamp(time).strftime("%Y-%m-%dT%H:%M") 
    
    if output not in ("vector", "raster"):
        raise ValueError(f"output must be 'vector' or 'raster', got {output!r}")
    
    url = (f'https://api.airtracker.createlab.org/get_footprint?view={lat},{lon},12&time={time_str}' 
           '&showContributionLikelihoods=true&isRealTime=true&format=geotiff')

    try:
        content = requests.get(url,timeout=60)
        content.raise_for_status()
        
        with MemoryFile(content.content) as memfile, memfile.open() as src:
            raster = rioxarray.open_rasterio(src).isel(band=3).load() # select band that shows whole footprint (not relative strength)

        if output == "raster":
            out = raster
        
        elif output == "vector":
            mask = raster.values > 0
            geoms = [shape(s) for s, _ in shapes(mask.astype("uint8"),mask=mask, transform=raster.rio.transform())]
            merged = unary_union(geoms) if geoms else None
            gdf = gpd.GeoDataFrame({"time": [time], "lon": [lon], "lat": [lat]},
                                   geometry=[merged], crs=raster.rio.crs).to_crs("EPSG:4326")
            out = gdf
    except Exception as e:
        warnings.warn(f"Footprint failed at {time_str}, {lat}, {lon}: {e}")
        out = None

    return out