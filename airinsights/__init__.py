"""
Analysis suite to translate local air quality data into actionable insights

Docs: https://github.com/edf-org/airinsights
Examples: https://github.com/edf-org/airinsights/tree/main/examples
"""

from .pollution_event_detection import pollution_event
from .helpers import read_aqdata_file,build_config,load_config