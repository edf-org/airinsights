import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import io

# session that retries with backoff after errors
session = requests.Session()
session.mount("https://", HTTPAdapter(pool_maxsize=10,max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))

BASE_URL = "https://esdr.cmucreatelab.org/api/v1"

def get_esdr(config_dict:dict,last_seen:dict = None,years_back: int = 5):
    """Fetch ESDR (CMU CREATE Lab) feed data, historical backfill + recent measurements, for feeds within a bounding box.

    Parameters
    ----------
    config_dict: dict
        Config dict from air.load_config that includes 'bounding_box' [min_lon, min_lat, max_lon, max_lat]
        and 'pollutant_map', whose values are the ESDR channel names to download (returned as columns)
    last_seen: dict
        Last stored timestamp (UTC) per feed_id, used to resume downloads.
        Feeds not in last_seen are backfilled.
    years_back: int
        How many years back to backfill data (defaults to 5).

    Returns:
        DataFrame in wide format, one row per feed and timestamp: datetime (UTC), feed_id, site_name,
        product_id, lat, lon, plus one column per channel. Empty if nothing new.
        site_name is the feed name followed by the feed_id in parentheses, so it is unique.
        product_id identifies the network (e.g. 101 PurpleAir, 122 AirMatters) for filtering later.
    """

    last_seen = last_seen or {}

    # get params from config
    if 'bounding_box' not in config_dict:
        raise KeyError("Config dict missing 'bounding_box' for area to download")
    bbox = tuple(config_dict['bounding_box'])
    pollutants = list(config_dict['pollutant_map'].values()) # esdr channel names, case-sensitive

    # get feeds and channels
    feeds = _esdr_metadata(bbox, pollutants)
    print(f"{len(feeds)} feeds found")

    now = datetime.now(timezone.utc)
    # unix seconds, 0 = full feed history
    backfill = 0 if years_back is None else (now - timedelta(days=365 * years_back)).timestamp()

    # fetch feeds in parallel
    def run(feed):
        try:
            return _esdr_feed(feed, last_seen, backfill, now)
        except requests.exceptions.RequestException:
            return None

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = [df for df in pool.map(run, feeds) if df is not None]

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def _esdr_metadata(bbox, pollutants):
    """ Page through public outdoor feeds in the bounding box that have a channel of interest """
    min_lon, min_lat, max_lon, max_lat = bbox
    params = {
        "whereAnd": f"latitude>={min_lat},latitude<={max_lat},longitude>={min_lon},longitude<={max_lon},exposure=outdoor",
        "fields": "id,name,productId,latitude,longitude,channelBounds",
        "limit": 200,
        "offset": 0,
    }
    rows = []
    while True:
        response = session.get(f"{BASE_URL}/feeds", params=params, timeout=60)
        response.raise_for_status()
        data = response.json()["data"]
        rows += data["rows"]
        params["offset"] += len(data["rows"])
        if not data["rows"] or params["offset"] >= data["totalCount"]:
            break

    feeds = []
    for row in rows:
        bounds = (row.get("channelBounds") or {}).get("channels", {}) # channels the feed has reported
        channels = [c for c in bounds if c in pollutants]
        if channels:
            row["channels"] = channels
            feeds.append(row)
    return feeds

def _esdr_feed(feed, last_seen, backfill, now):
    """ Fetch all new rows for one esdr feed """
    # start after data we already have, otherwise go back the full backfill window
    last_ts = last_seen.get(feed["id"])
    if last_ts is None or pd.isna(last_ts):
        from_ts = backfill
    else:
        from_ts = pd.to_datetime(last_ts, utc=True).timestamp() + 1

    url = f"{BASE_URL}/feeds/{feed['id']}/channels/{','.join(feed['channels'])}/export"
    try:
        response = session.get(url, params={"from": from_ts, "to": now.timestamp(), "format": "csv"})
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text)) # first column is unix time in seconds, converted to UTC below
    except (requests.exceptions.RequestException, pd.errors.EmptyDataError) as e:
        print(f"  skipping feed {feed['id']}: {e}")
        return None

    # export names columns like '3.feed_173351.PM2_5'; strip the prefix to leave the channel name
    df.columns = df.columns.str.replace(r"^\d+\.feed_\d+\.", "", regex=True)
    df = df.dropna(subset=feed["channels"], how="all") # drop rows where every channel is empty
    if df.empty:
        return None

    df["datetime"] = pd.to_datetime(df.iloc[:, 0], unit="s", utc=True).dt.round("ms") # round to remove sub-ms noise
    df = df.assign(feed_id=feed["id"], 
                        site_name=f"{feed['name'] or 'Unnamed'} ({feed['id']})", 
                        product_id=feed["productId"],
                        lat=feed["latitude"], 
                        lon=feed["longitude"])
    print(f"added {len(df)} new rows for feed {feed['id']}")
    return df[["datetime", "feed_id", "site_name", "product_id", "lat", "lon"] + feed["channels"]]