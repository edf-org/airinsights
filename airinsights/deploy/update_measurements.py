from ..download.get_openaq import get_openaq
from ..download.get_purpleair import get_purpleair

def get_meas(config_dict,last_seen):
    """Wrapper around data download scripts to download latest measurements for a specific data source"""
    if config_dict['data_source'] == 'openaq':
        df = get_openaq(config_dict,last_seen)
    elif config_dict['data_source'] == 'purpleair':
        df = get_purpleair(config_dict,last_seen)
    else:
        raise ValueError(
            f"Unknown data_source {config_dict['data_source']!r}. "
            "Valid options: 'openaq', 'purpleair'"
        )
    return df