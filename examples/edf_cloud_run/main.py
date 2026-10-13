import airinsights as air
from pathlib import Path
import os
import sys

# add a config file to the config folder to add to routine
config_dir = Path(__file__).parent / "config"

# function that triggers city_pipeline for each config using input from Cloud Run Jobs
def main(): 
    configs = sorted(config_dir.glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No configs found in {config_dir.resolve()}")
    
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", 0)) # when not provided task from GCP, runs 0 as default

    if task_index >= len(configs):
        print(f"Task {task_index}: only {len(configs)} configs exist, exiting cleanly")
        sys.exit(0)

    config_path = configs[task_index]
    print(f"Task {task_index}: running {config_path.name}")
    air.run_pipeline(config_file = config_path)
    
if __name__ == "__main__":
    main()