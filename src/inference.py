""" Module for model inference. """


import yaml
import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from train import TimeSeriesDataset, TSModel, MLP, available_models, device
import preprocess
import logging
logger = logging.getLogger("experiment_worker")

def descale(
        descaler,
        values
):
    values_2d = np.array(values)[:, np.newaxis]
    return descaler.inverse_transform(values_2d).flatten()


def predict(
        df,
        label_name,
        sequence_length,
        n_features,
        batch_size,
        base_path,
        model_id,
        model_name,
        scaler=None
):
    """Make predictions."""
    
    model_class = available_models.get(model_name)
    logger.info(f"USING MODEL CLASS {model_class}")
    if model_class is None:
        raise Exception(f"Unrecognized model_name: {model_name}. \nAvailable models: {list(available_models.keys())}")
    
    model = model_class(
        n_features=n_features,
        seq_len=sequence_length,
        batch_size=batch_size
    ).to(device)

    model.load_state_dict(torch.load(Path(base_path, f'model_{model_id}.pt')))
    model.eval()
    
    test_dataset = TimeSeriesDataset(np.array(df), np.array(df[label_name]), seq_len=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    labels = []
    with torch.no_grad():
        for features, target in test_loader:
            features = torch.Tensor(np.array(features))
            output = model(features)
            predictions.append(output.item())
            labels.append(target.item())

    # bring predictions back to original scale
    if scaler is None:
        scaler = joblib.load(Path(base_path, 'scaler.gz'))
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler.min_[0], scaler.scale_[0]
    predictions_descaled = descale(descaler, predictions)
    labels_descaled = descale(descaler, labels)

    return predictions_descaled, labels_descaled


def get_loss_metrics(
        y_true,
        y_pred,
):
    try:
        y_pred = y_pred[~np.isnan(y_pred)]
        y_true = y_true[~np.isnan(y_true)]

        rmse = round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 7)
        mae = round(metrics.mean_absolute_error(y_true, y_pred), 7)
        mape = round(metrics.mean_absolute_percentage_error(y_true, y_pred), 7)

        logger.info(f'RMSE: {rmse}')
        logger.info(f'MAE: {mae}')
        logger.info(f'MAPE: {mape}')
    except Exception as e:
        logger.info(e)
        logger.info(y_true)
        logger.info(y_pred)
        return 0, 0, 0

    return rmse, mae, mape