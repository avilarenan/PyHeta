from experiment import run_experiment
from experiment_params import experiments, exp_pattern
from tqdm.auto import tqdm
from config import base as basepath
import json
from pathlib import Path
from datetime import datetime, timezone
from time_utils import days_hours_minutes
import s3_utils
import pika
from time import sleep
import sys, os, json
import logging
import randomname
from FARM_LSTM.forecasting.scripts.static_info_utils import JSON_CREDENTIALS


worker_name = randomname.get_name(sep='_')

credentials = pika.PlainCredentials(JSON_CREDENTIALS["AMQP_USER"], JSON_CREDENTIALS["AMQP_PASSWORD"])


logger = logging.getLogger("experiment_worker")
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(f"../logs/experiment_worker_{worker_name}.log", mode="a", encoding="utf-8")
logger.addHandler(console_handler)
logger.addHandler(file_handler)
formatter = logging.Formatter(
   "{filename}:{lineno} - {funcName}|{asctime}[{levelname}]:{message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.setLevel("INFO")

AMQP_HOST = "10.193.254.138"

def callback_execute_experiment(channel, method, properties, body):
    delivery_tag = method.delivery_tag
    try:
        logger.info(f"[x] Received {body}")
        experiment_param = json.loads(body)
        ablation_id = experiment_param["ablation_id"]
        logger.info(f"Experiment params: {experiment_param}")
        base_folder = f"{basepath}/repos/FARM_LSTM/forecasting/experiments_{worker_name}/{ablation_id}"
        exp_pattern["ablation_id"] = ablation_id
        exp_pattern["s3_path"] = f"/{ablation_id}/"
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

        logger.info(f"""Uploading {base_folder}/.. to {exp_pattern["s3_bucket"]}""")
        s3_utils.upload_folder_to_s3(f"{base_folder}/..", exp_pattern["s3_bucket"])
        logger.info("Finished first uploading.")

        param_set = []
        
        params = experiment_param
        for i in tqdm(range(params["repetitions"])): # TODO: make a single repetition be the least acknowledgeable instruction
            list_of_features = params["FILTER_FEATURES"]
            name = f"""{params["ablation_id"]}_{'_'.join(list_of_features)}"""
            run_experiment(name, params, base_folder, i, R_scripts_folder, credentials_file)

        
        end_time = datetime.now(timezone.utc)
        time_log["end_time"] = f"{end_time}"
        time_log["time_spent"] = days_hours_minutes(end_time - start_time)
        time_log["total_number_of_experiments"] = len(param_set)
        time_log["avg_time_per_experiment"] = days_hours_minutes((end_time - start_time) / params["repetitions"])
        with open(f"{base_folder}/time.json", "w") as fp:
            json.dump(time_log, fp)

        channel.basic_ack(delivery_tag)
    except Exception as e:
        logger.exception(e)
        channel.basic_reject(delivery_tag)

def main():
    while True:
        try:
            params = pika.ConnectionParameters(heartbeat=12000, blocked_connection_timeout=12000, host=AMQP_HOST, credentials=credentials)
            connection = pika.BlockingConnection(params)
            
            channel = connection.channel()
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue='experiment_instructions', on_message_callback=callback_execute_experiment, auto_ack=False)

            logger.info(' [*] Waiting for messages. To exit press CTRL+C')
            channel.start_consuming()
        except pika.exceptions.StreamLostError as e:
            logger.exception(e)
            sleep(5)
            continue


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)