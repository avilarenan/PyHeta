import json

from config import working_base_path

JSON_CREDENTIALS = None
with open(f'{working_base_path}/../credentials.json') as f:
    JSON_CREDENTIALS = json.load(f)

AMQP_HOST = "10.193.254.138"

POSTGRES_HOST = "10.193.254.138"
POSTGRES_DB_DATA = "postgres"
POSTGRES_DB_TRAIN = "train"
POSTGRES_CONN_STRING_DATA = f'''postgresql://{JSON_CREDENTIALS["POSTGRESQL_USER"]}:{JSON_CREDENTIALS["POSTGRESQL_PASS"]}@{POSTGRES_HOST}/{POSTGRES_DB_DATA}'''
POSTGRES_CONN_STRING_TRAIN = f'''postgresql://{JSON_CREDENTIALS["POSTGRESQL_USER"]}:{JSON_CREDENTIALS["POSTGRESQL_PASS"]}@{POSTGRES_HOST}/{POSTGRES_DB_TRAIN}'''

PREVENT_DUPLICATE_TRAINING = False

EXOFEATURES_DATASET_MAPPING = {
    "FINANCE" : ["Open", "High", "Low", "Volume"],
    "ENERGY_LOAD" : ["p.amb.in.hp", "lux.west.lx", "lux.south.lx", "bat.load.w"],
    "ETTh1" : ["HUFL", "HULL", "LUFL", "LULL"],
    "METEOSUISSE" : ["temp.c", "wind.mps", "snow.cmsum", "precip.mmsum"],
}
    