import airinsights as air

# import config with openaq specifications 
config_dict = air.load_config("airinsights/config/detroit_openaq_config.yaml")

# requires environment variable OPENAQ_KEY for api authentication
# and config_dict with 'bounding_box' of area to download
df = air.get_openaq(config_dict,years_back=10,reference_only = True)

# write to file for analysis (compressed to save space on github)
df.to_csv('airinsights/examples/sample_data/detroit_reference_openaq_2016to2026.csv.gz',index=False)