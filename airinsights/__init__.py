"""
Analysis suite to translate local air quality data into actionable insights

Docs: https://github.com/edf-org/airinsights
Examples: https://github.com/edf-org/airinsights/tree/main/examples
"""

from .pollution_event_detection import pollution_event
from .pollution_event_classification import classify_pollution_events
from .helpers import read_aqdata_file,build_config,load_config
from .anomalous_sites import diurnal_hotspots
from .deploy.run_airinsights import run_airinsights
from .deploy.update_measurements import get_meas
from .download import get_openaq
from .download import get_purpleair
from .download import get_airtracker