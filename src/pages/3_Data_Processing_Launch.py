# streamlit_app.py

import streamlit as st
import datetime
import randomname
import pika
import json
from static_info_utils import JSON_CREDENTIALS, AMQP_HOST, EXOFEATURES_DATASET_MAPPING
import pandas as pd

credentials = pika.PlainCredentials(JSON_CREDENTIALS["AMQP_USER"], JSON_CREDENTIALS["AMQP_PASSWORD"])

st.set_page_config(
    page_title="DataProc Dashboard",
    page_icon="👋"
)

st.markdown("# DataProc Dashboard")
st.sidebar.header("Run DataProc")


st.markdown("### Launch a data processing step for an experiment")

available_datasets = ("ETTh1", "FINANCE", "ENERGY_LOAD", "METEOSUISSE")

selected_dataset = st.selectbox(
    "Dataset",
    available_datasets
)

selected_start_date = None
selected_end_date = None
selected_force_download_new_data = None
selected_ticker = None
selected_only_original_features = None
features = None

if selected_dataset == "FINANCE":
    selected_start_date = st.date_input("Start date", datetime.date(2020, 1, 1))
    selected_end_date = st.date_input("End date", datetime.date.today())

    selected_ticker = st.selectbox(
        "Ticker",
        ("PETR4", "VALE3", "ITUB4", "BPAC11", "B3SA3"),
        0
    )

    selected_target_feature = st.selectbox(
        "Target Feature",
        ("Open", "High", "Low", "Close"),
        3
    )

    selected_only_original_features = st.checkbox("ONLY_ORIGINAL_FEATURES", True)

    selected_force_download_new_data = st.selectbox(
        "Force Download New Data",
        (True, False),
        0
    )
elif selected_dataset == "ETTh1":
    selected_target_feature = st.selectbox(
        "Target Feature",
        ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"),
        6
    )
elif selected_dataset == "ENERGY_LOAD":
    selected_target_feature = st.selectbox(
        "Target Feature",
        ("pwr.prod.w","pwr.self.cons.w","pwr.tot.w","pwr.cons.w"),
        3
    )
elif selected_dataset == "METEOSUISSE":
    selected_target_feature = st.selectbox(
        "Target Feature",
        ("time","birch.npm3","precip.mmsum","rel.hum.pct","snow.cmsum","sun.minsum","temp.c","wind.mps"),
        1
    )
else:
    raise Exception(f"Unsupported dataset: {selected_dataset}. Valid values are: {available_datasets}")

selected_reuse_ceemdan = st.checkbox("REUSE CEEMDAN", True)
selected_force_new_prediction = st.checkbox("Force new prediction", True)
selected_skip_farm = st.checkbox("SKIP FARM", False)
selected_skip_correlation = st.checkbox("SKIP Correlation", False)
selected_skip_dtw = st.checkbox("SKIP DTW", True)
selected_only_shaped_features = st.checkbox("ONLY_SHAPED_FEATURES", False)
selected_only_target_feature_for_selected_imfs = st.checkbox("ONLY_TARGET_FEATURE_FOR_SELECTED_IMFS", True)

selected_only_target = st.checkbox("ONLY_TARGET", False)

selected_farm_threshold = st.number_input("Farm Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

st.divider()

option_single_or_multi = st.selectbox(
    "Single ablation test or Multiple ablation tests",
    ("Single","Multi"),
    1
)

if option_single_or_multi == "Single":

    selected_farm_fuzzyfy = st.checkbox("Farm Fuzzyfy", False)
    selected_farm_binary_shaping = st.checkbox("Farm binary shaping", False)
    selected_skip_ceemdan = st.checkbox("SKIP CEEMDAN", False)
    selected_skip_farm_shaping = st.checkbox("SKIP FARM SHAPING", False)
    selected_farm_ffalign = selected_farm_ffalign = st.checkbox("Farm FFAlign", False)

elif option_single_or_multi == "Multi":

    df = pd.DataFrame(
        [
            {
                "decomp": None,
                "locFarm win=5": None,
                "ffalign": None,
                "fuzzy": None,
                "bin_locRel": None
            },
        ]
    )
    selectbox_column_options = ["True", "False", "N/A"]
    edited_df = st.data_editor(df, num_rows="dynamic", column_config={
        "decomp": st.column_config.CheckboxColumn(
            "Decomposition",
            help="Whether to decompose time series with CEEMDAN or not",
            required=True,
            default=None
        ),
        "locFarm win=5": st.column_config.CheckboxColumn(
            "Farm shaping",
            help="Whether to reshape exogenous time series with FARM algorithm or not",
            required=True,
            default=None
        ),
        "ffalign": st.column_config.CheckboxColumn(
            "Forward Alignment",
            help="Use it or not, or does not apply when locFarm is disabled",
            default=None
        ),
        "fuzzy": st.column_config.CheckboxColumn(
            "Fuzzyfication",
            help="Use it or not, or does not apply when locFarm is disabled",
            default=None
        ),
        "bin_locRel": st.column_config.CheckboxColumn(
            "Binary Shaping",
            help="Binary or continuous shaping of exogenous features using FARM algorithm: Use it or not, or does not apply when locFarm is disabled",
            default=None
        ),
    },)

    features = st.multiselect(
        "Exogenous features to be considered in ablation",
        EXOFEATURES_DATASET_MAPPING[selected_dataset],
        EXOFEATURES_DATASET_MAPPING[selected_dataset]
    )

    st.text(f"Number of experiments: {edited_df.shape[0] * 2**len(features)}")

st.divider()

selected_train_frac = st.number_input("Train Fraction", min_value=0.0, max_value=1.0, value=0.7, step=0.01)

selected_repetions = st.number_input("Repetitions", min_value=1, max_value=5, value=1, step=1)

ablation_id = st.text_input("Ablation Id", randomname.get_name())
if st.button("New ablation id", type="primary"):
    ablation_id = randomname.get_name()

exp_pattern = {
    'ablation_id': ablation_id,
    'DATASET' : selected_dataset,
    'ticker' : selected_ticker,
    'start_date' : selected_start_date,
    'end_date' : selected_end_date,
    'target_feature' : selected_target_feature,
    'FORCE_DOWNLOAD_NEW_DATA': selected_force_download_new_data,
    'SKIP_CEEMDAN': None,
    'REUSE_CEEMDAN': selected_reuse_ceemdan,
    'FORCE_NEW_PREDICTION': selected_force_new_prediction,
    'SKIP_FARM': selected_skip_farm,
    'SKIP_FARM_SHAPING': None,
    'SKIP_CORRELATION' : selected_skip_correlation,
    'SKIP_DTW': selected_skip_dtw,
    'ONLY_SHAPED_FEATURES': selected_only_shaped_features,
    'ONLY_TARGET_FEATURE_FOR_SELECTED_IMFS': selected_only_target_feature_for_selected_imfs,
    'ONLY_ORIGINAL_FEATURES': selected_only_original_features,
    'ONLY_TARGET': selected_only_target,
    'farm_threshold': selected_farm_threshold,
    'farm_ffalign' : None,
    'farm_fuzzyfy' : None,
    'farm_binary_shaping': None,
    'latter_imfs' : ["IMF3", "IMF4", "IMF5", "IMF6", "IMF7", "IMF8", "IMF9", "Residue"],
    'initial_imfs' : ["IMF0", "IMF1", "IMF2"],
    'FILTER_FEATURES' : [],
    'ablation_features' : [],
    'train_frac' : selected_train_frac,
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
    'repetitions' : int(selected_repetions),
    'save_files_to_s3' : True,
    's3_path' : f"/{ablation_id}/",
    's3_bucket' : "forecastingexperimentsdata",
}

st.write("Selected ablation pattern:")
st.json(exp_pattern)

if st.button("Launch data processing", type="primary"):

    if exp_pattern["DATASET"] in EXOFEATURES_DATASET_MAPPING:
        if features is None:
            features = EXOFEATURES_DATASET_MAPPING[exp_pattern["DATASET"]]
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

    print(lists_of_features)

    ablations = {}

    if option_single_or_multi == "Single":
        print("SINGLE")
        ablations[ablation_id] = {}
        for list_of_features in lists_of_features:
            exp_params = exp_pattern.copy()
            exp_params["SKIP_CEEMDAN"] = selected_skip_ceemdan
            exp_params["SKIP_FARM_SHAPING"] = selected_skip_farm_shaping
            exp_params["farm_ffalign"] = selected_farm_ffalign
            exp_params["farm_fuzzyfy"] = selected_farm_fuzzyfy
            exp_params["farm_binary_shaping"] = selected_farm_binary_shaping
            exp_params["FILTER_FEATURES"] = list_of_features
            ablations[ablation_id][f"exp_filtered_{'_'.join(list_of_features)}"] = exp_params

    elif option_single_or_multi == "Multi":
        print("MULTI")
        experiments_variable_parameters = edited_df.to_records(index=False)
        for row in experiments_variable_parameters:
            ablation_id = randomname.get_name()
            print(f"{ablation_id} : {row}")
            for list_of_features in lists_of_features:
                exp_params = exp_pattern.copy()
                exp_params["ablation_id"] = ablation_id
                exp_params["SKIP_CEEMDAN"] = not row[0]
                exp_params["SKIP_FARM_SHAPING"] = not row[1]
                exp_params["farm_ffalign"] = row[2]
                exp_params["farm_fuzzyfy"] = row[3]
                exp_params["farm_binary_shaping"] = row[4]
                exp_params["FILTER_FEATURES"] = list_of_features
                if ablation_id not in ablations:
                    ablations[ablation_id] = {}
                ablations[ablation_id][f"exp_filtered_{'_'.join(list_of_features)}"] = exp_params
    else:
        raise Exception(f"Unexpected option_single_or_multi = {option_single_or_multi}")

    st.json(ablations)

    connection = pika.BlockingConnection(pika.ConnectionParameters(AMQP_HOST, credentials=credentials))
    channel = connection.channel()
    for ablationid, values in ablations.items():
        print(f"ablation_id = {ablation_id}")
        for name, instruction in values.items():
            print(f"name = {name}")
            print(f"instruction = {instruction}")
            st.json(instruction)
            channel.basic_publish(
                exchange='',
                routing_key='dataproc_instructions',
                body=json.dumps(instruction, default=str)
            )
    connection.close()