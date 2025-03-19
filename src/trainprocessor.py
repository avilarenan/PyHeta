import os
import json
import boto3
from sqlalchemy import create_engine
import pandas as pd
import uuid
from datetime import datetime
import joblib
from pathlib import Path
import plotly.express as px
from pathlib import Path
from static_info_utils import POSTGRES_CONN_STRING_TRAIN, POSTGRES_CONN_STRING_DATA, PREVENT_DUPLICATE_TRAINING
import logging
import gzip
from io import BytesIO
import ast
logger = logging.getLogger("train_worker")

pd.options.plotting.backend = "plotly"

client = boto3.client('s3')

dbengine_train = create_engine(POSTGRES_CONN_STRING_TRAIN)
# dbengine_train = dbengine_train.connect()

dbengine_data = create_engine(POSTGRES_CONN_STRING_DATA)
# dbengine_data = engine_data.connect()

import train, inference, interpret, s3_utils, reset_experiments

def run_train(experiment_name, experiment_params, base_folder, run_number):

    params = experiment_params

    target_feature_name = params["target_feature"]
    Windows = params["windows"]
    dataset_id = params["dataset_id"]

    df_list = []
    query = f""" SELECT * FROM public."Result_metrics" """
    for chunk_dataframe in pd.read_sql(query, dbengine_train, chunksize=50000):
        df_list += [chunk_dataframe]
    df_results_ref = pd.concat(df_list)
    already_processed_datasets = list(df_results_ref["dataset_id"].unique())

    if dataset_id in already_processed_datasets and PREVENT_DUPLICATE_TRAINING:
        run_train_id = list(df_results_ref[df_results_ref["dataset_id"] == dataset_id]["run_train_unique_id"].unique())
        logger.info(f"dataset_id {dataset_id} already passed through training and evaluation process: run_train_unique_id = {run_train_id}.")
        logger.info(f"preventing duplicated training")
        return

    run_train_unique_id = str(uuid.uuid4())
    logger.info(f"Beginning training {experiment_name} | run_train_unique_id = {run_train_unique_id}")
    experiment_name = experiment_name + f".{run_number}"
    Path(os.path.join(base_folder, experiment_name)).mkdir(parents=True, exist_ok=True)
    base_folder = f"{base_folder}/{experiment_name}"

    Path(base_folder + "/output/step_1_raw_data").mkdir(parents=True, exist_ok=True)
    Path(base_folder + "/output/step_2_additional_data").mkdir(parents=True, exist_ok=True)
    Path(base_folder + "/output/step_3_ceemdan_data/decomposition").mkdir(parents=True, exist_ok=True)
    Path(base_folder + "/output/step_3_ceemdan_data/grouped_imfs").mkdir(parents=True, exist_ok=True)
    Path(base_folder + "/output/step_4_farm_data/farm_shaped_ceemdan_grouped_imfs").mkdir(parents=True, exist_ok=True)
    Path(base_folder + "/output/step_5_final_preprocessed_data").mkdir(parents=True, exist_ok=True)
    Path(base_folder + "/output/step_6_training_phase_data/error_metrics_result").mkdir(parents=True, exist_ok=True)
    Path(base_folder + "/output/step_7_prediction_data/imfs_prediction").mkdir(parents=True, exist_ok=True)
    Path(base_folder + "/output/step_8_shap_inference/shap_result").mkdir(parents=True, exist_ok=True)
    Path(base_folder + "/output/step_9_recomposition_data/final_result").mkdir(parents=True, exist_ok=True)

    df_list = []
    query = f""" SELECT * FROM public."Datasets" where dataset_id = '{dataset_id}' """
    for chunk_dataframe in pd.read_sql(query, dbengine_data, chunksize=50000):
        df_list += [chunk_dataframe]

    df_datasets_ref = pd.concat(df_list)

    dataset_name = list(df_datasets_ref["dataset"].unique())[0]
    SKIP_FARM_SHAPING = list(df_datasets_ref["skip_farm_shaping"].unique())[0]

    df_train_per_imf = {}
    df_test_per_imf = {}

    train_datasets = list(df_datasets_ref[df_datasets_ref["dataset_type"] == "train"]["unique_id"])
    test_datasets = list(df_datasets_ref[df_datasets_ref["dataset_type"] == "test"]["unique_id"])
    scaler_s3_path = None
    bucket_name = None
    for train_dataset_unique_id, test_dataset_unique_id in zip(train_datasets, test_datasets):
        df_list = []
        query = f""" SELECT * FROM public."{train_dataset_unique_id}" """
        for chunk_dataframe in pd.read_sql(query, dbengine_data, chunksize=50000):
            df_list += [chunk_dataframe]
        df_current_train_dataset = pd.concat(df_list)

        df_list = []
        query = f""" SELECT * FROM public."{test_dataset_unique_id}" """
        for chunk_dataframe in pd.read_sql(query, dbengine_data, chunksize=50000):
            df_list += [chunk_dataframe]
        df_current_test_dataset = pd.concat(df_list)

        possible_imfs = list(df_datasets_ref["component"].unique())
        for imf in possible_imfs:
            filtered_train_df = df_current_train_dataset[df_current_train_dataset["component"] == imf]
            filtered_test_df = df_current_test_dataset[df_current_test_dataset["component"] == imf]

            if filtered_train_df.empty or filtered_test_df.empty:
                continue
            
            df_train_per_imf[imf] = filtered_train_df.copy()
            df_test_per_imf[imf] = filtered_test_df.copy()
        
        # Retrieving scaler saved at dataprocessing step - these values should be constant over unique datasets beloging to a single dataset id
        df_rf_filtered_by_unique_id = df_datasets_ref[df_datasets_ref["unique_id"] == train_dataset_unique_id]
        scaler_paths = list(df_rf_filtered_by_unique_id["scaler_s3_path"].unique())
        basefolders = list(df_rf_filtered_by_unique_id["base_folder"].unique())
        bucket_names = list(df_rf_filtered_by_unique_id["s3_bucket"].unique())

        if len(scaler_paths) > 1:
            raise Exception(f"Unexpected length for scaler paths = {len(scaler_paths)} - {basefolders}. Expected 1.")
        if len(basefolders) > 1:
            raise Exception(f"Unexpected length for basefolders = {len(basefolders)} - {basefolders}. Expected 1.")
        if len(basefolders) > 1:
            raise Exception(f"Unexpected length for bucket_names = {len(bucket_names)} - {bucket_names}. Expected 1.")
        
        scaler_path = scaler_paths[0]
        basefolder = basefolders[0]
        bucket_name = bucket_names[0]

        scaler_s3_path = f"""{"".join(scaler_path.split("/")[1:])}/{basefolder.split("/")[-1]}/scaler.gz"""

    # Retrieving scaler saved at dataprocessing step
    logger.info(f"Retrieving scaler from: {bucket_name} | {scaler_s3_path}")
    scaler_obj = client.get_object(Bucket=bucket_name, Key=scaler_s3_path)
    body = scaler_obj['Body']
    gzipfile = BytesIO(body.read())
    scaler = joblib.load(gzipfile)

    possible_imfs = list(df_train_per_imf.keys())
    logger.info(f"possible_imfs = {possible_imfs}")
    logger.info(f"df_train_per_imf = {df_train_per_imf}")
    logger.info(f"df_test_per_imf = {df_test_per_imf}")
    for component in possible_imfs:
        logger.info(f"Training and predicting {component} of {target_feature_name}")
        
        ### DIVISION

        ### TRAIN LSTM

        train_df = df_train_per_imf[component]
        test_df = df_test_per_imf[component]

        sequence_length = params["sequence_length"]
        if component in Windows:
            sequence_length = Windows[component]

        batch_size = params["batch_size"]
        n_epochs = params["n_epochs"]
        n_epochs_stop = params["n_epochs_stop"]
        min_delta_stop = params["min_delta_stop"]
        learning_rate = params["learning_rate"]
        label_name = params["target_feature"]

        actual_columns = train_df.columns.to_list()

        # Optionally select only original features or only shaped features if wanted for testing
        ONLY_TARGET = params["ONLY_TARGET"]

        if not ONLY_TARGET:
            ONLY_ORIGINAL_FEATURES = params["ONLY_ORIGINAL_FEATURES"]
            if ONLY_ORIGINAL_FEATURES:
                actual_columns = params["FILTER_FEATURES"] + [target_feature_name] + ["Noise"]
                
                train_df = train_df[actual_columns]
                test_df = test_df[actual_columns]

            ONLY_SHAPED_FEATURES = params["ONLY_SHAPED_FEATURES"]
            if ONLY_SHAPED_FEATURES:
                if SKIP_FARM_SHAPING:
                    raise Exception("Cannot use ONLY_SHAPED_FEATURES = True when also SKIP_FARM_SHAPING = True. Please change params accordingly.")
                filter_features = params["FILTER_FEATURES"] + [target_feature_name]
                filter_features.remove(target_feature_name)
                shaped_features = [f"{element}_shaped" for element in filter_features] + ["Noise"] + [target_feature_name]
                
                try:
                    train_df = train_df[shaped_features]
                    test_df = test_df[shaped_features]
                except Exception as e:
                    logger.info(e)
                    logger.info(shaped_features)
                    logger.info(train_df.columns)
                    logger.info(test_df.columns)

        filter_features_param = params["FILTER_FEATURES"].copy()

        ONLY_TARGET_FEATURE_FOR_SELECTED_IMFS = params["ONLY_TARGET_FEATURE_FOR_SELECTED_IMFS"] # suggested to be true since higher level IMFs is simple and thus do not exogenous data
        IMFs_to_predict_with_only_target_feature = params["latter_imfs"]
        if ONLY_TARGET:
            IMFs_to_predict_with_only_target_feature += params["initial_imfs"]
            ONLY_SHAPED_FEATURES = True
            ONLY_ORIGINAL_FEATURES = True
        
        if ONLY_TARGET_FEATURE_FOR_SELECTED_IMFS == True and component in IMFs_to_predict_with_only_target_feature:
            train_df = train_df[[target_feature_name]]
            test_df = test_df[[target_feature_name]]
            filter_features_param = [target_feature_name]

        if filter_features_param is not None:
            if isinstance(filter_features_param, str):
                filter_features_param = ast.literal_eval(filter_features_param)
            elif isinstance(filter_features_param, list):
                logger.debug("Filter features param is list")
            else:
                raise Exception(f"Unexpected type {type(filter_features_param)}. Expected ones can be list or list as string")
            if ''in filter_features_param:
                filter_features_param.remove('')
            logger.info(f"filter_features_param = {filter_features_param}")
            filter_features_param += [target_feature_name]
            filter_features_param = list(set(filter_features_param))
            train_df = train_df[filter_features_param]
            test_df = test_df[filter_features_param]

        if params["SKIP_CEEMDAN"] is False:
            n_epochs = params["n_epochs_by_imf"][component]
            n_epochs_stop = params["n_epochs_stop_by_imf"][component]
            learning_rate = params["learning_rate_by_imf"][component]
            model_name = params["model_by_imf"][component]
            
        else:
            model_name = params["default_model"]
        
        model_id = f"component_{component}"

        logger.info(train_df)
        logger.info(test_df)

        hist = train.train_model(
            train_df,
            test_df,
            label_name,
            sequence_length,
            batch_size,
            n_epochs,
            n_epochs_stop,
            learning_rate,
            min_delta_stop,
            base_folder,
            model_name,
            model_id
        )

        fig = hist.plot(title="Training and test losses Vs Epochs")
        fig.write_html(f"{base_folder}/output/step_6_training_phase_data/{component}_training_curves.html")

        ### Evaluate the model

        test_predictions_descaled, test_labels_descaled = inference.predict(
            df=test_df,
            label_name=label_name,
            sequence_length=sequence_length,
            n_features=test_df.shape[1],
            batch_size=batch_size,
            base_path=base_folder,
            model_id=model_id,
            model_name=model_name,
            scaler=scaler
        )

        train_predictions_descaled, train_labels_descaled = inference.predict(
            df=train_df,
            label_name=label_name,
            sequence_length=sequence_length,
            n_features=train_df.shape[1],
            batch_size=batch_size,
            base_path=base_folder,
            model_id=model_id,
            model_name=model_name,
            scaler=scaler
        )

        logger.info(f'Error on all test data for {component}:')
        rmse_test, mae_test, mape_test = inference.get_loss_metrics(test_labels_descaled, test_predictions_descaled)
        rmse_train, mae_train, mape_train = inference.get_loss_metrics(train_labels_descaled, train_predictions_descaled)
        pd.DataFrame({
            "rmse_test" : [rmse_test],
            "mae_test" : [mae_test],
            "mape_test" : [mape_test],
            "rmse_train" : [rmse_train],
            "mae_train" : [mae_train],
            "mape_train" : [mape_train],
        }).to_csv(f"{base_folder}/output/step_6_training_phase_data/error_metrics_result/{component}_error_metrics.csv")
        
        pd.DataFrame().from_records([{
            "run_train_unique_id" : run_train_unique_id,
            "rmse_test" : rmse_test,
            "mae_test" : mae_test,
            "mape_test" : mape_test,
            "rmse_train" : rmse_train,
            "mae_train" : mae_train,
            "mape_train" : mape_train,
            "component" : component,
            "dataset_id" : dataset_id
        }], index="run_train_unique_id").to_sql(f'step_6_component_error_metrics', con=dbengine_train, method=None, schema="public", if_exists='append', index=True)

        # plot predictions vs true values
        df_pred_test = pd.DataFrame()
        df_pred_test['test_predicted'] = test_predictions_descaled
        df_pred_test['test_true'] = test_labels_descaled
        df_pred_test['test_residual'] = test_labels_descaled - test_predictions_descaled

        df_pred_train = pd.DataFrame()
        df_pred_train['train_predicted'] = train_predictions_descaled
        df_pred_train['train_true'] = train_labels_descaled
        df_pred_train['train_residual'] = train_labels_descaled - train_predictions_descaled

        fig = px.line(df_pred_test, title="Predicted Vs. True on Test dataset for target feature")
        df_pred_test.to_csv(f"{base_folder}/output/step_7_prediction_data/imfs_prediction/{component}_test_predicted.csv")
        df_pred_test["run_train_unique_id"] = run_train_unique_id
        df_pred_test["component"] = component
        df_pred_test["type"] = "test"
        df_pred_test["dataset_id"] = dataset_id

        fig.write_html(f"{base_folder}/output/step_7_prediction_data/imfs_prediction/{component}_test_predicted_vs_true.html")

        fig = px.line(df_pred_train, title="Predicted Vs. True on Train dataset for target feature")
        df_pred_train.to_csv(f"{base_folder}/output/step_7_prediction_data/imfs_prediction/{component}_train_predicted.csv")
        df_pred_train["run_train_unique_id"] = run_train_unique_id
        df_pred_train["component"] = component
        df_pred_train["type"] = "train"
        df_pred_train["dataset_id"] = dataset_id

        pd.concat([df_pred_train, df_pred_test]).to_sql(f'step_7_prediction', con=dbengine_train, method=None, schema="public", if_exists='append', index=True)

        fig.write_html(f"{base_folder}/output/step_7_prediction_data/imfs_prediction/{component}_train_predicted_vs_true.html")

        ### Interpretability and explainability

        if ONLY_TARGET_FEATURE_FOR_SELECTED_IMFS is True and component in IMFs_to_predict_with_only_target_feature:
            continue # skip multiple features interpretability if only using 1 feature as input

        background_data_size = 900
        test_sample_size = 100

        filter_features_param = params["FILTER_FEATURES"].copy()

        if filter_features_param is not None:
            actual_columns = [target_feature_name] + filter_features_param
       
        # calculate permutation feature importance instead of shap feature importance, it seems shap does not work with lstm


    ## CEEMDAN recomposition and metrics

    predicted_recomposition_train = None
    predicted_recomposition_test = None
    prediction_path = f"{base_folder}/output/step_7_prediction_data/imfs_prediction/"

    for file in os.listdir(os.fsencode(prediction_path)):
        filename = os.fsdecode(file)
        logger.info(filename)
        if not filename.endswith(".csv"):
            continue
        df_predicted = pd.read_csv(prediction_path + filename)

        if "train" in filename:
            if predicted_recomposition_train is None:
                predicted_recomposition_train = df_predicted
            else:
                predicted_recomposition_train += df_predicted

        if "test" in filename:
            if predicted_recomposition_test is None:
                predicted_recomposition_test = df_predicted
            else:
                predicted_recomposition_test += df_predicted

    predicted_recomposition_test.to_csv(f"{base_folder}/output/step_9_recomposition_data/final_result/final_prediction_test.csv")
    fig = px.line(predicted_recomposition_test, title="Final prediction test",)
    fig.write_html(f"{base_folder}/output/step_9_recomposition_data/final_result/final_prediction_test.html")
    predicted_recomposition_test["type"] = "test"
    predicted_recomposition_test["run_train_unique_id"] = run_train_unique_id
    predicted_recomposition_test["dataset_id"] = dataset_id

    predicted_recomposition_train.to_csv(f"{base_folder}/output/step_9_recomposition_data/final_result/final_prediction_train.csv")
    fig = px.line(predicted_recomposition_train, title="Final prediction train",)
    fig.write_html(f"{base_folder}/output/step_9_recomposition_data/final_result/final_prediction_train.html")
    predicted_recomposition_train["type"] = "train"
    predicted_recomposition_train["run_train_unique_id"] = run_train_unique_id
    predicted_recomposition_train["dataset_id"] = dataset_id

    pd.concat([predicted_recomposition_train, predicted_recomposition_test]).to_sql(f'step_9_recomposition', con=dbengine_train, method=None, schema="public", if_exists='append', index=True)


    rmse_test, mae_test, mape_test = inference.get_loss_metrics(
        predicted_recomposition_test["test_predicted"],
        predicted_recomposition_test["test_true"],
    )

    rmse_train, mae_train, mape_train = inference.get_loss_metrics(
        predicted_recomposition_train["train_predicted"],
        predicted_recomposition_train["train_true"],
    )

    pd.DataFrame({
        "final_rmse_test" : [rmse_test],
        "final_mae_test" : [mae_test],
        "final_mape_test" : [mape_test],
        "final_rmse_train" : [rmse_train],
        "final_mae_train" : [mae_train],
        "final_mape_train" : [mape_train],
    }).to_csv(
        f"{base_folder}/output/step_9_recomposition_data/final_result/final_error_metrics.csv"
    )
    

    pd.DataFrame().from_records([{
        'run_train_unique_id': run_train_unique_id,
        'dataset_id': dataset_id,
        'dataset_type': dataset_name,
        'farm_fuzzyfy': params["farm_fuzzyfy"],
        'farm_binary_shaping': params["farm_binary_shaping"],
        'farm_ffalign': params["farm_ffalign"],
        'skip_ceemdan': params["SKIP_CEEMDAN"],
        'skip_farm_shaping': params["SKIP_FARM_SHAPING"],
        'dataset': params["DATASET"],
        'features': params["FILTER_FEATURES"],
        'ablation_id': params["ablation_id"],
        'target_feature': target_feature_name,
        'created_at': datetime.now(),
        'base_folder' : base_folder,
        's3_bucket' : params["s3_bucket"],
        'final_rmse_test' : rmse_test,
        'final_mae_test' : mae_test,
        'final_mape_test' : mape_test,
        'final_rmse_train' : rmse_train,
        'final_mae_train' : mae_train,
        'final_mape_train' : mape_train,
    }], index="run_train_unique_id").to_sql(f'Result_metrics', con=dbengine_train, method=None, schema="public", if_exists='append', index=True)

    with open(f"{base_folder}/params.json", "w") as fp:
        json.dump(params, fp)

    if params["save_files_to_s3"] is True:
        logger.info(f"""Uploading files to S3: from {base_folder} to {params["s3_bucket"]}""")
        s3_utils.upload_folder_to_s3(f"{base_folder}/..", params["s3_bucket"], params["ablation_id"])
        logger.info(f"Clearing folder: {base_folder}/..")
        reset_experiments.clear_folder(f"{base_folder}/..")

    logger.info(f"Finished experiment {experiment_name}")