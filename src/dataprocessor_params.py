from config import base as basepath
from config import credentials_file

params = {
    "DATASET": "FINANCE",
    'ticker' : 'PETR4',
    'start_date' : "2020-01-01",
    'end_date' : "2024-01-01",
}

target_feature = {
    "ETTh1" : "OT",
    "FINANCE" : "Close",
    "ENERGY_LOAD" : "pwr.cons.w",
    "METEOSUISSE" : "birch.npm3"
}

params["target_feature"] = target_feature[params["DATASET"]]

base_folder = f"{basepath}/repos/FARM_LSTM/forecasting/tmp/"