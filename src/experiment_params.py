import randomname
from datetime import datetime, timezone

exp_pattern = {
    'ablation_id': None,
    'DATASET' : "ETTh1", # "FINANCE" or "ENERGY_LOAD" or "ETTh1"
    'ticker' : 'PETR4',
    'start_date' : "2020-01-01",
    'end_date' : "2024-01-01",
    # 'target_feature' : 'Close', # FINANCE
    # 'target_feature' : 'pwr.cons.w', # ENERGY_LOAD # pwr.prod.w,pwr.self.cons.w,pwr.tot.w
    'target_feature' : 'OT', # ETTh1 # date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
    'FORCE_DOWNLOAD_NEW_DATA': True,
    'SKIP_CEEMDAN': False,
    'REUSE_CEEMDAN': True,
    'FORCE_NEW_PREDICTION': True,
    'SKIP_FARM': False,
    'SKIP_FARM_SHAPING': False,
    'SKIP_CORRELATION' : False,
    'SKIP_DTW': True,
    'ONLY_SHAPED_FEATURES': False,
    'ONLY_TARGET_FEATURE_FOR_SELECTED_IMFS': True,
    'ONLY_ORIGINAL_FEATURES': True,
    'ONLY_TARGET': False,
    'farm_threshold': 0.5, #
    'latter_imfs' : ["IMF3", "IMF4", "IMF5", "IMF6", "IMF7", "IMF8", "IMF9", "Residue"],
    'initial_imfs' : ["IMF0", "IMF1", "IMF2"],
    'FILTER_FEATURES' : [],
    'ablation_features' : [],
    'train_frac' : 0.7,
    'sequence_length' : 8,
    'windows': {
        "IMF0" : 2,
        "IMF1" : 3,
        "IMF2" : 4,
        "IMF3" : 4,
        "IMF4" : 5,
        "Residue" : 6,
    },
    'learning_rate' : 0.001,
    'learning_rate_by_imf' : {
        "IMF0": 0.0010,
        "IMF1": 0.0010,
        "IMF2": 0.0010,
        "IMF3": 0.0010,
        "IMF4": 0.0010,
        "Residue": 0.0001
    },
    'batch_size' : 64,
    'n_epochs' : 500,
    'n_epochs_by_imf' : {
        "IMF0": 1000,
        "IMF1": 1000,
        "IMF2": 1000,
        "IMF3": 500,
        "IMF4": 500,
        "Residue": 10000
    },
    'n_epochs_stop' : 5,
    'n_epochs_stop_by_imf' : {
        "IMF0": 5,
        "IMF1": 10,
        "IMF2": 10,
        "IMF3": 10,
        "IMF4": 10,
        "Residue": 50
    },
    'model_by_imf' : {
        "IMF0": "LSTM",
        "IMF1": "LSTM",
        "IMF2": "LSTM",
        "IMF3": "LSTM",
        "IMF4": "LSTM",
        "Residue": "MLP"
    },
    'default_model' : "LSTM",
    'min_delta_stop': 0.0000,
    'background_data_size' : 900,
    'test_sample_size' : 100,
    'repetitions' : 3,
    'save_files_to_s3' : True,
    's3_path' : "/",
    's3_bucket' : "forecastingexperiments",
}

if exp_pattern["DATASET"] == "FINANCE":
    features = [ # FINANCE DATASET
        "Open", "High", "Low", "Volume", # "Day_Of_Week", "Month_Of_Year", "Quarter_Of_Year", "High_Low_Pct", "Open_Close_Pct", "Noise"
    ]
elif exp_pattern["DATASET"] == "ENERGY":
    features = [ # ENERGY DATASET
        "p.amb.in.hp", "lux.west.lx", "lux.south.lx", "bat.load.w"
    ]
elif exp_pattern["DATASET"] == "ETTh1":
    features = [ # ETTh1 DATASET
        "HUFL", "HULL", "LUFL", "LULL"
    ]
elif exp_pattern["DATASET"] == "METEOSUISSE":
    features = [ # Air metrics DATASET
        "rel.hum.pct", "temp.c","wind.mps", "snow.cmsum"
    ]
else:
    raise Exception(f"""Unexpected {exp_pattern["DATASET"]} Dataset""")

exp_pattern["ablation_features"] = features

# features = [] # NOTE: comment if you wish all combinations of exogenous features to be experimented.

from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

lists_of_features = []

for i, combo in enumerate(powerset(features), 1):
    if len(combo) >= 1:
        lists_of_features += [list(combo)]

lists_of_features += [[]]

experiments = {}

for list_of_features in lists_of_features:
    exp_params = exp_pattern.copy()
    exp_params["FILTER_FEATURES"] = list_of_features

    experiments[f"exp_filtered_{'_'.join(list_of_features)}"] = exp_params