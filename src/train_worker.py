from trainprocessor import run_train
from experiment_params import experiments, exp_pattern
from tqdm.auto import tqdm
from config import base as basepath, working_base_path as wbasepath
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
import threading
import functools
from static_info_utils import JSON_CREDENTIALS, AMQP_HOST


worker_name = randomname.get_name(sep='_')

credentials = pika.PlainCredentials(JSON_CREDENTIALS["AMQP_USER"], JSON_CREDENTIALS["AMQP_PASSWORD"])


logger = logging.getLogger("train_worker")
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(f"{wbasepath}/../logs/train_worker_{worker_name}.log", mode="a", encoding="utf-8")
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

def ack_message(ch, delivery_tag):
    """Note that `ch` must be the same pika channel instance via which
    the message being ACKed was retrieved (AMQP protocol constraint).
    """
    if ch.is_open:
        ch.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        logger.error("Channel is already closed, something wrong has happened.")

def on_message(ch, method_frame, _header_frame, body, args):
    connection, thrds = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(ch, delivery_tag, body))
    t.start()
    thrds.append(t)

def do_work(ch, delivery_tag, body):
    thread_id = threading.get_ident()
    logger.info(f'Thread id: {thread_id} Delivery tag: {delivery_tag} Message body: {body}')

    try:
        logger.info(f"[x] Received {body}")
        experiment_param = json.loads(body)
        ablation_id = experiment_param["ablation_id"]
        logger.info(f"Experiment params: {experiment_param}")
        base_folder = f"{basepath}/repos/FARM_LSTM/forecasting/experiments_{worker_name}/{ablation_id}"
        exp_pattern["ablation_id"] = ablation_id
        exp_pattern["s3_path"] = f"/{ablation_id}/"
        Path(base_folder).mkdir(parents=True, exist_ok=True)

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
            run_train(name, params, base_folder, i)
        
        end_time = datetime.now(timezone.utc)
        time_log["end_time"] = f"{end_time}"
        time_log["time_spent"] = days_hours_minutes(end_time - start_time)
        time_log["total_number_of_experiments"] = len(param_set)
        time_log["avg_time_per_experiment"] = days_hours_minutes((end_time - start_time) / params["repetitions"])
        with open(f"{base_folder}/time.json", "w") as fp:
            json.dump(time_log, fp)
        
    except Exception as e:
        logger.exception(e)
        ch.basic_publish(
            exchange='',
            routing_key='train_backout',
            body=json.dumps(body.decode('utf-8'), default=str)
        )

    cb = functools.partial(ack_message, ch, delivery_tag)
    ch.connection.add_callback_threadsafe(cb)

def main():
    while True:
        try:
            threads = []
            params = pika.ConnectionParameters(heartbeat=10, blocked_connection_timeout=65535, host=AMQP_HOST, credentials=credentials)
            connection = pika.BlockingConnection(params)
            
            channel = connection.channel()
            channel.basic_qos(prefetch_count=1)
            on_message_callback = functools.partial(on_message, args=(connection, threads))
            channel.basic_consume(queue='train_instructions', on_message_callback=on_message_callback, auto_ack=False)

            logger.info(' [*] Waiting for messages. To exit press CTRL+C')
            channel.start_consuming()

            for thread in threads:
                thread.join()
            
            connection.close()
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