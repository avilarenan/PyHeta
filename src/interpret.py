""" Module for model interpretation. """


import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import shap
import traceback 
import torch
from torch.utils.data import DataLoader
from train import TimeSeriesDataset, TSModel, MLP, available_models, device
import preprocess
import logging
logger = logging.getLogger("experiment_worker")

def get_important_features(
        train_df,
        test_df,
        background_data_size,
        test_sample_size,
        sequence_length,
        n_features,
        batch_size,
        base_path,
        model_id,
        model_name,
        filter_columns=None,
        target_feature_name="Close",
):
    try:
        # load data
        train_df = train_df.astype(np.float32)
        test_df = test_df.astype(np.float32)
        
        logger.info(filter_columns)
        if filter_columns is not None:
            train_df = train_df[filter_columns]
            test_df = test_df[filter_columns]

        if "Date" in train_df.columns:
            train_df = train_df.set_index("Date")
        if "Date" in test_df.columns:
            test_df = test_df.set_index("Date")
        logger.info(train_df)
        
        # load trained model
        model_class = available_models.get(model_name)
        logger.info(f"USING MODEL CLASS {model_class}")
        if model_class is None:
            raise Exception(f"Unrecognized model_name: {model_name}. \nAvailable models: {list(available_models.keys())}")
        n_features = train_df.shape[1]
        model = model_class(
            n_features=n_features,
            seq_len=sequence_length,
            batch_size=batch_size
        ).to(device)

        model.load_state_dict(torch.load(Path(base_path, f'model_{model_id}.pt')))
        
        model.eval()

        # get background dataset
        train_dataset = TimeSeriesDataset(np.array(train_df), np.array(train_df[target_feature_name]), seq_len=sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=background_data_size, shuffle=False)
        background_data, _ = next(iter(train_loader))

        # get test data samples on which to explain the model’s output
        test_dataset = TimeSeriesDataset(np.array(test_df), np.array(test_df[target_feature_name]), seq_len=sequence_length)
        test_loader = DataLoader(test_dataset, batch_size=test_sample_size, shuffle=False)
        test_sample_data, _ = next(iter(test_loader))

        # integrate out feature importances based on background dataset
        ## TODO: try matching the sequence length in order to  converge SHAP
        output = model(torch.Tensor(np.array(background_data)))
        output_test = model(torch.Tensor(np.array(test_sample_data)))
        e = shap.DeepExplainer(model, torch.Tensor(np.array(background_data)))

        # explain the model's outputs on some data samples
        shap_values = e.shap_values(torch.Tensor(np.array(test_sample_data)))
        shap_values = np.absolute(shap_values)
        shap_values = np.mean(shap_values, axis=0)

        ret = shap_values.squeeze()

        return ret
    except Exception as e:
        logger.exception(f"Error during SHAP explaining: {e}")
        traceback.print_exc()