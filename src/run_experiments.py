from experiment import run_experiment
from experiment_params import experiments, ablation_id, exp_pattern
from tqdm.auto import tqdm
import multiprocessing as mp
from config import base as basepath
import json
from pathlib import Path
from datetime import datetime, timezone
from time_utils import days_hours_minutes
import s3_utils

if __name__ == '__main__':

    base_folder = f"{basepath}/repos/FARM_LSTM/forecasting/experiments/{ablation_id}"
    Path(base_folder).mkdir(parents=True, exist_ok=True)

    R_scripts_folder = f"{basepath}/repos/FARM_LSTM/forecasting/farm"
    credentials_file = f"{basepath}/repos/FARM_LSTM/forecasting/credentials.json"

    with open(f"{base_folder}/ablation_pattern.json", "w") as fp:
        json.dump(exp_pattern, fp)

    with open(f"{base_folder}/experiments.json", "w") as fp:
        json.dump(experiments, fp)

    start_time = datetime.now(timezone.utc)
    time_log = {"start_time" : f"{start_time}"}
    with open(f"{base_folder}/time.json", "w") as fp:
        json.dump(time_log, fp)

    print(f"""Uploading {base_folder}/.. to {exp_pattern["s3_bucket"]}""")
    s3_utils.upload_folder_to_s3(f"{base_folder}/..", exp_pattern["s3_bucket"])
    print("Finished first uploading.")

    param_set = []
    
    mp.set_start_method('spawn')

    for name, params in tqdm(experiments.items()):
        for i in tqdm(range(params["repetitions"])):
            param_set += [(name, params, base_folder, i, R_scripts_folder, credentials_file)]

    param_set = [param_set[0]]

    with mp.Pool(1) as pool:
        pool.starmap(
            run_experiment,
            param_set
        )
    
    end_time = datetime.now(timezone.utc)
    time_log["end_time"] = f"{end_time}"
    time_log["time_spent"] = days_hours_minutes(end_time - start_time)
    time_log["total_number_of_experiments"] = len(param_set)
    time_log["avg_time_per_experiment"] = days_hours_minutes((end_time - start_time) / len(param_set))
    with open(f"{base_folder}/time.json", "w") as fp:
        json.dump(time_log, fp)