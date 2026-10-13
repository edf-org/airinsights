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