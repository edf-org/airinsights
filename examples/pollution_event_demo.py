import airinsights as air
import seaborn as sns
import matplotlib.pyplot as plt

# --- Set path to data source ---
input_data = "sample_data/oakland_2017_100x100_blackcarbon.csv"

# --- Provide config details for your AQ data ---

# Option 1: set path to config file
config_path = "../config/100x100_config.yaml"

# Option 2: generate new config for your data
#config_path = helpers.build_config(
#    timestamp_col = 'Datetime',
#    site_col = 'Site',
#    value_col = 'hourly_BC',
#    datetime_format= "%m/%d/%Y %H:%M",
#    lat_col= 'Latitude',
#    lon_col= 'Longitude',
#    config_file_path = '../config/test_config_output.yaml'
#)

# --- Read in data and config ---
df, config = air.read_aqdata_file(input_data,config = config_path)

# --- Run function ---
events = air.pollution_event(df,config,verbose=True)
print(events)

# --- Join back to full df ---
df = df.merge(events,on = [config['site_col'],config['timestamp_col']],how="left")

# Optional; write to csv
# result.to_csv('oakland_2017_100x100_blackcarbon_pollution_events.csv', index = False)

# --- Visualize flagged events for a single monitoring site ---

# Select data for one site
plot_data = df[df[config['site_col']] == df[config['site_col']].iloc[0]]

# Generate timeseries plot of all measurements
sns.lineplot(data=plot_data, 
             x=config['timestamp_col'], 
             y=config['value_col'],
             color="gray",
             zorder=1)

# Add points for two types of pollution events
sns.scatterplot(
    data=plot_data[plot_data['event_type'].isin(['Unusually high','Extremely high'])], 
    x=config['timestamp_col'], 
    y=config['value_col'], 
    hue="event_type", 
    palette=['orange', 'red'],
    s=100, 
    zorder=2                 
)
plt.title(f"Pollution event timeseries for site {plot_data[config['site_col']].unique().item()}")
plt.show()