<div align="center">

<img src="https://www.edf.org/themes/edf2020/images/source/site-logo.svg" height="100"/> 

## **Air Insights**
### Turn your city’s air quality data into insights that drive health and climate action

</div>
For over a decade, EDF scientists have worked with partners around the world to develop new ways of analyzing air quality data to meet targeted needs including (but not limited to) tracking pollution hotspots, identifying local sources, developing customized policy interventions, and measuring health impacts.
<br>
<br>
We are now honing and automating these methods to develop <b>Air Insights</b> — a powerful suite of automated, open-source analyses designed to help users process large volumes of air quality measurements and provide answers to common, yet complex, questions like:<br><br>

* How severe is air pollution in my area? 
* Are certain locations or times more impacted?
* What are the likely sources contributing to air pollution hotspots?
* How has air quality changed over time? 
* Are air quality improvement policies in my area working?
* Are there areas that urgently require follow-up investigation or health alerts?
  
  <br>
---
<div align="center">

# Beta Release 2026.10.XX

</div>

## 💡 Core Features

* **Event Detection** – Detect and flag measurements that are atypical for each monitor at different hours of the day with the `pollution_event()` function
* **Trend Analysis** – Quantify diurnal, seasonal, and long-term trends, controlling for weather variability to help assess changes over time such as from policies like low emission zones or fuel restrictions with the `annual_trends()` function
* **Source Identification** – Identify likely locations of upwind pollution sources using Air Tracker with the `source_area()` function 
* **Areas of Interest** – Detect locations that repeatedly show unusual pollution patterns compared to the monitoring network as a whole with the `anomalous_site()` function 


## 🕑 Coming Soon

* **Data Audit** – Assess AQ data quality, completeness, and reliability

<div align="center">

</div>

<hr>

## 📖 Documentation

Access documentation for each function using Python in your IDE of choice.

For example:
```r
help(airinsights)
help(pollution_event)
```

<hr>

## 🗃️ Installation

The source code is currently hosted on GitHub at:
https://github.com/edf-org/airinsights

Binary installers for the latest released version are available at the [Python Package Index (PyPI)](https://pypi.org/project/airinsights).

```sh
# PyPI
pip install airinsights
```

<hr>

## 📖 Examples

Example scripts using sample data are located in the [examples](https://github.com/edf-org/airinsights/tree/main/examples) folder on GitHub:
* [Pollution event detection example](https://github.com/edf-org/airinsights/blob/main/examples/pollution_event_demo.py)
* [Trend analysis example]
* [Areas of interest example]
* [Source area example] 

Air Insights uses a configuration file to ensure your data can be parsed correctly. Set up instructions are also located in the [examples](https://github.com/edf-org/airinsights/tree/main/examples) folder. 
* [Setting Up Air Insights]


<hr>

## License
**Air Insights** is licensed under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
