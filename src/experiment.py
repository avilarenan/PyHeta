import os
import json
import subprocess
import os
import pandas as pd
from pathlib import Path
from PyEMD import CEEMDAN
import plotly.express as px
from pathlib import Path
from dtw import dtw
import scipy.stats
import logging
logger = logging.getLogger("experiment_worker")

pd.options.plotting.backend = "plotly"

import preprocess, train, inference, interpret, get_raw_data, s3_utils, reset_experiments

def run_experiment(experiment_name, experiment_params, base_folder, run_number, R_scripts_folder, credentials_file=None):
    logger.info(f"Beginning experiment {experiment_name}")
    experiment_name = experiment_name + f".{run_number}"
    Path(os.path.join(base_folder, experiment_name)).mkdir(parents=True, exist_ok=True)
    original_base_folder = base_folder
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

    params = experiment_params

    target_feature_name = params["target_feature"]

    ## DATA
    if params["DATASET"] == "FINANCE":
        ### Download Raw Data
        ticker = params["ticker"] + ".SA"
        start_date = params["start_date"]
        end_date = params["end_date"]
        filename = f"raw_data"

        FORCE_DOWNLOAD_NEW_DATA = params["FORCE_DOWNLOAD_NEW_DATA"]
        if FORCE_DOWNLOAD_NEW_DATA:

            fig = (get_raw_data
                .download_financial_raw_data_single_solutions_dataservices(ticker, start_date, end_date, filename, f"{base_folder}/output/step_1_raw_data/", credentials_path=credentials_file)
                .plot(title=f"Raw Data {ticker}"))
            fig.write_html(f"{base_folder}/output/step_1_raw_data/original.html")

        ### Prepare basic new features
        file_name = f"{filename}.csv" # raw_data
        data_raw = pd.read_csv(f"{base_folder}/output/step_1_raw_data/{file_name}")
        data = data_raw

        train_df, test_df = preprocess.prep_data_basic(
            df=data_raw,
            train_frac=params["train_frac"],
            keep_date=True,
            scale_features=False,
            base_path=base_folder,
            feature_to_base_noise=target_feature_name
        )

        df = pd.concat([train_df, test_df])
        df.to_csv(f"{base_folder}/output/step_2_additional_data/preprocessed_basicfeatured_data.csv")
        fig = df.set_index("Date").plot(title="Basic featured data")
        fig.write_html(f"{base_folder}/output/step_2_additional_data/basic_featured.html")

        logger.info("Prepared basic featured data")
    
    elif params["DATASET"] == "ENERGY_LOAD":
        logger.info(f"CWD: {os.getcwd()}")
        filename = f"../energy_load_data/hslu.rc.data.large.clean.csv"
        energy_data = pd.read_csv(filename)
        energy_data = energy_data.rename({"date" : "Date"}, axis=1)
        data_raw = energy_data
        energy_data = energy_data.set_index("Date")
        fig = energy_data.plot(title=f"Raw Data Energy Load")
        fig.write_html(f"{base_folder}/output/step_1_raw_data/original.html")

        ### Prepare basic new features | no features to prepare for energy load data
        train_df, test_df = preprocess.prep_data_basic(
            df=data_raw,
            train_frac=params["train_frac"],
            keep_date=True,
            scale_features=False,
            base_path=base_folder,
            feature_to_base_noise=target_feature_name
        )
        df = pd.concat([train_df, test_df])
        df.to_csv(f"{base_folder}/output/step_2_additional_data/preprocessed_basicfeatured_data.csv")
        df = df.reset_index()
        fig = df.set_index("Date").plot(title="Basic featured data")
        fig.write_html(f"{base_folder}/output/step_2_additional_data/basic_featured.html")

        logger.info("Prepared basic featured data")

    elif  params["DATASET"] == "ETTh1": # ETTh1 DataSet : https://github.com/zhouhaoyi/ETDataset => Target: OL (oil temperature) | paper DOI: 10.1007/978-3-031-50396-2_14
        filename = f"../ETTh1_dataset/ETTh1.csv"
        ETTh1_data = pd.read_csv(filename)
        ETTh1_data = ETTh1_data.rename({"date" : "Date"}, axis=1)
        data_raw = ETTh1_data
        ETTh1_data = ETTh1_data.set_index("Date")
        fig = ETTh1_data.plot(title=f"Raw Data ETTh1")
        fig.write_html(f"{base_folder}/output/step_1_raw_data/original.html")

        ### Prepare basic new features | no features to prepare for ETTh1 load data
        train_df, test_df = preprocess.prep_data_basic(
            df=data_raw,
            train_frac=params["train_frac"],
            keep_date=True,
            scale_features=False,
            base_path=base_folder,
            feature_to_base_noise=target_feature_name
        )
        df = pd.concat([train_df, test_df])
        df.to_csv(f"{base_folder}/output/step_2_additional_data/preprocessed_basicfeatured_data.csv")
        df = df.reset_index()
        fig = df.set_index("Date").plot(title="Basic featured data")
        fig.write_html(f"{base_folder}/output/step_2_additional_data/basic_featured.html")

        logger.info("Prepared basic featured data")
    elif params["DATASET"] == "METEOSUISSE":
        # columns: "time","birch.npm3","precip.mmsum","rel.hum.pct","snow.cmsum","sun.minsum","temp.c","wind.mps"
        filename = f"../METEOSUISSE_dataset/swiss.meteo.data.csv"
        meteo_data = pd.read_csv(filename)
        meteo_data = meteo_data.rename({"time" : "Date"}, axis=1)
        data_raw = meteo_data
        meteo_data = meteo_data.set_index("Date")
        fig = meteo_data.plot(title=f"Raw Data METEOSUISSE")
        fig.write_html(f"{base_folder}/output/step_1_raw_data/original.html")

        ### Prepare basic new features | no features to prepare for ETTh1 load data
        train_df, test_df = preprocess.prep_data_basic(
            df=data_raw,
            train_frac=params["train_frac"],
            keep_date=True,
            scale_features=False,
            base_path=base_folder,
            feature_to_base_noise=target_feature_name
        )
        df = pd.concat([train_df, test_df])
        df.to_csv(f"{base_folder}/output/step_2_additional_data/preprocessed_basicfeatured_data.csv")
        df = df.reset_index()
        fig = df.set_index("Date").plot(title="Basic featured data")
        fig.write_html(f"{base_folder}/output/step_2_additional_data/basic_featured.html")

        logger.info("Prepared basic featured data")
    
    else :
        raise Exception(f"Dataset : {params['DATASET']} Not implemented")

    ### CEEMDAN
    # TODO: Fix CEEMDAN : https://www.inf.ufpr.br/lesoliveira/download/ESWA2017.pdf
    SKIP_CEEMDAN = params["SKIP_CEEMDAN"]
    REUSE_CEEMDAN = params["REUSE_CEEMDAN"]

    if not SKIP_CEEMDAN:
        could_not_reuse_ceemdan = False
        if REUSE_CEEMDAN:
            logger.info("Reusing last CEEMDAN, supposing it exists.")
            decompositions = {}
            for feature in df.set_index("Date").columns:
                try:
                    if feature not in params["FILTER_FEATURES"] + [target_feature_name]:
                        continue
                    decomposed_signal_df = pd.read_csv(f"{base_folder}/output/step_3_ceemdan_data/decomposition/{feature}_CEEMDAN_Decomposition.csv")
                    logger.info(f"Reusing last CEEMDAN for feature {feature}.")
                    decomposed_signal_df.index.name = feature
                    # decomposed_signal_df.columns = [
                    #     f"IMF{column}" if int(column) != len(decomposed_signal_df.columns) - 1 else "Residue" for column in decomposed_signal_df.columns
                    # ]
                    decompositions[feature] = decomposed_signal_df
                except Exception as e:
                    logger.info(f"Redecomposing with CEEMDAN from scratch because of a problem trying to reuse CEEMDAN for {feature}: \n{e}")
                    could_not_reuse_ceemdan = True
                    break
        if not REUSE_CEEMDAN or could_not_reuse_ceemdan:
            logger.info("Beginning new ceemdan decomposition")
            ceemdan = CEEMDAN(trials=5, parallel=False, processes=2)
            
            # Decomposing
            decompositions = {}
            for feature in df.set_index("Date").columns:
                if feature not in params["FILTER_FEATURES"] + [target_feature_name]:
                    continue
                logger.info(f"Decomposing {feature}")
                signal = df[feature].to_numpy()
                c_imfs = ceemdan(signal, max_imf=len(params["initial_imfs"])+1, progress=True)
                decomposed_signal_df = pd.DataFrame(c_imfs).T
                decomposed_signal_df.columns = [
                    f"IMF{column}" if int(column) != len(decomposed_signal_df.columns) - 1 else "Residue" for column in decomposed_signal_df.columns
                ]
                decomposed_signal_df.index.name = feature
                decomposed_signal_df.to_csv(
                    f"{base_folder}/output/step_3_ceemdan_data/decomposition/{feature}_CEEMDAN_Decomposition.csv"
                )
                decompositions[feature] = decomposed_signal_df

    else:
        # NOTE: here we do not decompose
        # We just make the original dataframe in the same output format as of a decomposed one, so that the further steps keep working
        # We are calling the original series as IMF0 single component decomposition just for sake of simplicity
        logger.info("Not decomposing with ceemdan.")
        decompositions = {}
        for feature in df.set_index("Date").columns:
            signal_df = pd.DataFrame()
            signal_df["IMF0"] = df[feature]
            decompositions[feature] = signal_df
            
            dummy_df = pd.DataFrame(columns=['IMF0'])
            dummy_df.index.name = feature
            dummy_df.to_csv(
                f"{base_folder}/output/step_3_ceemdan_data/decomposition/{feature}_CEEMDAN_Decomposition.csv"
            )
            
    # Grouping by IMF
    imf_grouped_decompositions = {}
    for key, value in decompositions.items():
        for column in value.columns.to_list():
            if column not in imf_grouped_decompositions:
                imf_grouped_decompositions[column] = []
            imf_grouped_decompositions[column] += [value[column].rename(f"{key}")]

    for key, value in imf_grouped_decompositions.items():
        imf_grouped_decompositions[key] = pd.concat(value, axis=1)
        imf_grouped_decompositions[key].to_csv(
            f"{base_folder}/output/step_3_ceemdan_data/grouped_imfs/{key}_CEEMDAN.csv"
        )
        

        logger.info("Finished decomposing with CEEMDAN.")

    ### FARM

    SKIP_FARM = params["SKIP_FARM"]
    SKIP_FARM_SHAPING = params["SKIP_FARM_SHAPING"]
    if not SKIP_FARM:
        logger.info("FARM Shaping")
        farm_input_data_path = f"{base_folder}/output/step_3_ceemdan_data/grouped_imfs/"
        farm_output_data_path = f"{base_folder}/output/step_4_farm_data/farm_shaped_ceemdan_grouped_imfs/"

        farm_threshold = params["farm_threshold"]
        farm_ffalign = params["farm_ffalign"]
        farm_fuzzyc = params["farm_fuzzyfy"]
        farm_binary_shaping = params["farm_binary_shaping"]
        subprocess.call(
            f"/usr/bin/Rscript --vanilla {R_scripts_folder}/farm_shape.R {farm_input_data_path} {farm_output_data_path} {farm_threshold} {target_feature_name} {farm_ffalign} {farm_fuzzyc} {farm_binary_shaping}",
            shell=True
        )

        logger.info("Finished FARM")

    SKIP_CORRELATION = params["SKIP_CORRELATION"]
    if not SKIP_CORRELATION:
        corr_input_data_path = f"{base_folder}/output/step_3_ceemdan_data/grouped_imfs/"
        corr_output_data_path = f"{base_folder}/output/step_4_farm_data/farm_shaped_ceemdan_grouped_imfs/"
        for file in os.listdir(os.fsencode(corr_input_data_path)):
            filename = os.fsdecode(file)
            logger.info(f"{filename} corr")
            if not filename.endswith(".csv"):
                continue
            df_corr_input = pd.read_csv(corr_input_data_path + filename)
            correlations = {}

            for feature in df_corr_input.columns:
                if "index" in feature:
                    continue
                x = df_corr_input[feature].to_numpy()
                if target_feature_name not in df_corr_input.columns:
                    continue
                y = df_corr_input[target_feature_name].to_numpy()
                corr, _ = scipy.stats.pearsonr(x, y)
                correlations[feature] = [corr]
            pd.DataFrame().from_dict(correlations).to_csv(f"{corr_output_data_path}/{filename}_corr.csv")
        

    SKIP_DTW = params["SKIP_DTW"]
    if not SKIP_DTW:
        dtw_input_data_path = f"{base_folder}/output/step_3_ceemdan_data/grouped_imfs/"
        dtw_output_data_path = f"{base_folder}/output/step_4_farm_data/farm_shaped_ceemdan_grouped_imfs/"
        for file in os.listdir(os.fsencode(dtw_input_data_path)):
            filename = os.fsdecode(file)
            logger.info(f"{filename} DTW")
            if not filename.endswith(".csv"):
                continue
            logger.info(f"DTWing")
            df_dtw_input = pd.read_csv(dtw_input_data_path + filename)
            dtw_alignments = {}
            for feature in df_dtw_input.columns:
                alignment = dtw(
                    df_dtw_input[feature].to_numpy(),
                    df_dtw_input[target_feature_name].to_numpy(),
                    # keep_internals=True
                )
                dtw_alignments[feature] = [alignment.distance]
            pd.DataFrame().from_dict(dtw_alignments).to_csv(f"{dtw_output_data_path}/{filename}_dtw.csv")

    ### Prediction
    FORCE_NEW_PREDICTION = params["FORCE_NEW_PREDICTION"]
    Windows = params["windows"]

    possible_imfs_df = pd.read_csv(f"{base_folder}/output/step_3_ceemdan_data/decomposition/{target_feature_name}_CEEMDAN_Decomposition.csv")
    possible_imfs = possible_imfs_df.drop(target_feature_name, axis=1).columns.to_list()

    if FORCE_NEW_PREDICTION:

        logger.info(f"possible_imfs = {possible_imfs}")
        for component in possible_imfs:
            logger.info(f"Training and predicting {component} of {target_feature_name}")

            ### Fullfeatured FARM Shaped Data Load
            file_name_before_farm = f"{base_folder}/output/step_3_ceemdan_data/grouped_imfs/{component}_CEEMDAN.csv"

            if not SKIP_FARM_SHAPING:
                file_name_after_farm = f"{base_folder}/output/step_4_farm_data/farm_shaped_ceemdan_grouped_imfs/farm_shaped_{component}_CEEMDAN.csv"
            else:
                file_name_after_farm = file_name_before_farm
            
            data_after_farm = pd.read_csv(file_name_after_farm)
            data_before_farm = pd.read_csv(file_name_before_farm)

            data_before_farm = data_before_farm.loc[:, ~data_before_farm.columns.str.contains('^Unnamed')]
            data_before_farm = data_before_farm.loc[:, ~data_before_farm.columns.str.contains('^X')]
            data_after_farm = data_after_farm.loc[:, ~data_after_farm.columns.str.contains('^Unnamed')]
            data_after_farm = data_after_farm.loc[:, ~data_after_farm.columns.str.contains('^X')]

            logger.info(data_raw.columns)
            date_series = data_raw["Date"] ## TODO: LCWIN date forcedly fixed -> do it in R

            data_after_farm["Date"] = date_series
            data_before_farm["Date"] = date_series
            data_after_farm = data_after_farm.set_index("Date")
            data_before_farm = data_before_farm.set_index("Date")

            data = pd.concat([data_before_farm, data_after_farm], axis=1)
            data = data.loc[:, ~data.columns.duplicated()]
            data = data.reset_index()

            if params["DATASET"] in ["FINANCE", "ENERGY_LOAD", "ETTh1", "METEOSUISSE"]:
                train_df, test_df = preprocess.prep_data_basic(
                    df=data,
                    train_frac=params["train_frac"],
                    keep_date=True,
                    add_noise=True,
                    base_path=base_folder,
                    feature_to_base_noise=target_feature_name
                )

            else:
                raise Exception(f"""Unexpected dataset {params["DATASET"]}""")

            df = pd.concat([train_df, test_df])
            df.to_csv(
                f"{base_folder}/output/step_5_final_preprocessed_data/{component}_final_preprocessed_data.csv"
            )

            train_df = train_df.drop(["Date"], axis=1)
            test_df = test_df.drop(["Date"], axis=1)

            ### Explore Final Preprocessed Data

            final_df = pd.read_csv(
                f'{base_folder}/output/step_5_final_preprocessed_data/{component}_final_preprocessed_data.csv'
            )
            final_df = final_df.drop(final_df.columns[:1].tolist(), axis=1)

            final_df = final_df.set_index("Date")
            fig = px.line(final_df, title="Features Vs Time (days)")
            fig.add_vline(final_df.index[len(train_df)], line_width=3, line_dash="dash", line_color="green")
            fig.add_vrect(x0=final_df.index[len(train_df)], x1=final_df.index[-1], 
                        annotation_text="Test dataset", annotation_position="top left",
                        fillcolor="green", opacity=0.25, line_width=0)
            fig.write_html(f"{base_folder}/output/step_5_final_preprocessed_data/{component}_basic_featured_plus_farm_and_noise.html")

            ### TRAIN LSTM

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
                logger.info(filter_features_param)
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
                model_name=model_name
            )

            train_predictions_descaled, train_labels_descaled = inference.predict(
                df=train_df,
                label_name=label_name,
                sequence_length=sequence_length,
                n_features=train_df.shape[1],
                batch_size=batch_size,
                base_path=base_folder,
                model_id=model_id,
                model_name=model_name
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
            fig.write_html(f"{base_folder}/output/step_7_prediction_data/imfs_prediction/{component}_test_predicted_vs_true.html")

            fig = px.line(df_pred_train, title="Predicted Vs. True on Train dataset for target feature")
            df_pred_train.to_csv(f"{base_folder}/output/step_7_prediction_data/imfs_prediction/{component}_train_predicted.csv")
            fig.write_html(f"{base_folder}/output/step_7_prediction_data/imfs_prediction/{component}_train_predicted_vs_true.html")

            ### Interpretability and explainability

            if ONLY_TARGET_FEATURE_FOR_SELECTED_IMFS is True and component in IMFs_to_predict_with_only_target_feature:
                continue # skip multiple features interpretability if only using 1 feature as input

            background_data_size = 900
            test_sample_size = 100

            filter_features_param = params["FILTER_FEATURES"].copy()

            if filter_features_param is not None:
                actual_columns = [target_feature_name] + filter_features_param
            try:
                shap_values = interpret.get_important_features(
                    background_data_size,
                    test_sample_size,
                    filter_columns=actual_columns,
                    sequence_length=sequence_length,
                    n_features=train_df.shape[1],
                    batch_size=batch_size,
                    base_path=base_folder,
                    model_id=model_id,
                    model_name=model_name,
                    target_feature_name=params["target_feature"],
                )

                shap_plot = pd.DataFrame(shap_values, columns=train_df.columns.tolist())
                shap_plot['days'] = [i-sequence_length+1 for i in list(range(1,sequence_length+1))]

                shap_plot.to_csv(f"{base_folder}/output/step_8_shap_inference/shap_result/{component}_shap_metrics.csv")

                fig = px.line(shap_plot.set_index("days"), title="SHAP Feature importance", )
                fig.write_html(f"{base_folder}/output/step_8_shap_inference/shap_result/{component}_shap.html")
            except AssertionError as e:
                logger.info(f"Error while getting the shapley results: {e}. Moving on without it.")


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

    predicted_recomposition_train.to_csv(f"{base_folder}/output/step_9_recomposition_data/final_result/final_prediction_train.csv")
    fig = px.line(predicted_recomposition_train, title="Final prediction train",)
    fig.write_html(f"{base_folder}/output/step_9_recomposition_data/final_result/final_prediction_train.html")


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

    with open(f"{base_folder}/params.json", "w") as fp:
        json.dump(params, fp)

    if params["save_files_to_s3"] is True:
        logger.info(f"""Uploading files to S3: from {base_folder} to {params["s3_bucket"]}""")
        s3_utils.upload_folder_to_s3(f"{base_folder}/..", params["s3_bucket"], params["ablation_id"])
        logger.info(f"Clearing folder: {base_folder}/..")
        reset_experiments.clear_folder(f"{base_folder}/..")

    logger.info(f"Finished experiment {experiment_name}")