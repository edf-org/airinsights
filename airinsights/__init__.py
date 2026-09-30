"""
Analysis suite to translate local air quality data into actionable insights

Docs: https://github.com/edf-org/airinsights
Examples: https://github.com/edf-org/airinsights/tree/main/examples
"""

from .pollution_events import pollution_events
from .annual_trends import annual_trends
from .helpers import read_aqdata_file,build_config,load_config,STANDARD_POLLUTANTS
from .anomalous_sites import anomalous_sites
from .source_area import source_area