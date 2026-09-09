import airinsights as air
import seaborn as sns
import matplotlib.pyplot as plt

df,config = air.read_aqdata_file("airinsights/examples/sample_data/detroit_reference_openaq_2016to2026.csv.gz",
                                 config = "airinsights/config/detroit_openaq_config.yaml")

# run on no2, pm2.5 and so2
df = df[df['parameter'].isin(['no2','so2','pm25'])]

# convert units to ppb for interpretation
is_ppm = df["units"].str.lower() == "ppm"
df["value"] = df["value"].where(~is_ppm, df["value"] * 1000)
df["units"] = df["units"].where(~is_ppm, "ppb")

network, annual, monthly, data = air.site_trends(df,config,return_data=True)

# first, plot the network level result for one pollutant
network_data = data['network_monthly']
network_data = network_data[network_data['parameter'] == "no2"] # select a pollutant
plt.plot(network_data.index, network_data.value, marker='o', label='Monthly mean')
plt.plot(network_data.index, network_data.trend_line, linestyle='--', label='Trend')
plt.legend()
plt.show()

# now look at heatmap of trends by site and pollutant
trend = annual.pivot(index="site_name", columns="parameter", values="slope")
sig = annual.pivot(index="site_name", columns="parameter", values="p_value") < 0.05
labels = trend.round(1).astype(str) + sig.replace({True: " *", False: ""})
vmax = trend.abs().max().max()
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(trend, annot=labels, fmt="", cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                cbar_kws={"label": "trend"}, linewidths=1, linecolor="white", ax=ax)
ax.set_title("Annual trend by site (* = significant, p < 0.05)")
ax.set_ylabel("")
fig.tight_layout()
plt.show()

# look at trends by month for a specific site
site = "Detroit-Southwest"
site_of_interest = monthly[monthly['site_name'] == site]
trend = site_of_interest.pivot(index="parameter", columns="month", values="slope")
sig = site_of_interest.pivot(index="parameter", columns="month", values="p_value") < 0.05
labels = trend.round(1).astype(str) + sig.map(lambda x: ' *' if x else '')
vmax = trend.abs().max().max()
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(trend, annot=labels, fmt="", cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
            cbar_kws={"label": "trend"}, linewidths=1, linecolor="white", ax=ax)
ax.set_title(f"Monthly trend at {site} (* = significant, p < 0.05)")
ax.set_ylabel("")
ax.set_xlabel("")
fig.tight_layout()
plt.show()

# and check trend line for specific site and pollutant
site_data = data['site_monthly']
site_data = site_data[(site_data['site_name'] == site) & (site_data['parameter'] == "pm25")]
plt.plot(site_data.index, site_data.value, marker='o', label='Monthly mean')
plt.plot(site_data.index, site_data.trend_line, linestyle='--', label='Trend')
plt.legend()
plt.show()
