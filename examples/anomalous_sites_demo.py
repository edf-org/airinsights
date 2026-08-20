import airinsights as air
from airinsights.anomalous_sites import anomalous_sites
import matplotlib.pyplot as plt
import seaborn as sns

# --- Set path to data source ---
input_data = "examples/sample_data/oakland_2017_100x100_blackcarbon.csv"

# --- Provide config details for your AQ data ---

# Option 1: set path to config file
config_path = "config/100x100_config.yaml"

# Option 2: generate new config for your data
#config_path = air.build_config(
#    timestamp_col = 'Datetime',
#    site_col = 'Site',
#    value_col = 'hourly_BC',
#    pollutant_col = 'pollutant',
#    datetime_format= "%m/%d/%Y %H:%M",
#    lat_col= 'Latitude',
#    lon_col= 'Longitude',
#    config_file_path = '../config/test_config_output.yaml'
#)

# --- Read in data and config ---
df, config = air.read_aqdata_file(input_data,config = config_path)

# --- Run function ---
sites = anomalous_sites(df,config)
print(sites)

# --- Get sites with unusual patterns compared to the network for the 30 day timeframe ---
anomalous_sites_30d = (
    sites.loc[sites.timeframe == '30d']
    .groupby('Site')
    .agg({'n_hours_elevated': 'first','times_elevated': 'first', 'site_mean': 'mean'})
)
anomalous_sites_30d = (
    anomalous_sites_30d.loc[anomalous_sites_30d.times_elevated != 'None']
    .rename(
        columns = {
            'n_hours_elevated': '# of elevated hours', 
            'times_elevated': 'Times elevated', 
            'site_mean': 'Average BC (ug/m3)'
        }
    )
)
print(anomalous_sites_30d)

# Optional; write to csv
# anomalous_sites_30d.to_csv('oakland_2017_100x100_blackcarbon_anomalous_sites_30d.csv')

# --- Plot network median and site mean with anamolous hours highlighted ---
sites_30d = sites.loc[sites.timeframe == '30d']
example_site = 8
example_site_data = (
    (sites_30d[sites_30d.Site == example_site])[['hour', 'site_mean', 'network_median']]
    .rename(
        columns = {
            'hour': 'Hour of day',
            'site_mean': 'Site mean',
            'network_median': 'Network median'
        }
    )
    .set_index('Hour of day')
)

flagged_hours = (
    sites_30d[(sites_30d.Site == example_site) & sites_30d.elevated][['hour', 'site_mean', 'network_median']]
    .rename(
            columns = {
                'hour': 'Hour of day',
                'site_mean': 'Flagged hours',
            }
        )
)


sns.set_theme(style="whitegrid")
ax = sns.lineplot(data = example_site_data, palette = ['red', 'blue'], dashes = False)
sns.scatterplot(
    data=flagged_hours, 
    x='Hour of day', 
    y='Flagged hours', 
    s = 100,
    color = 'purple', 
    label = "Flagged hours"
)
sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
)
ax.set_ylabel('BC (ug/m3)')
plt.show()