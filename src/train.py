""" Module for model training. """


import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
logger = logging.getLogger("experiment_worker")

import preprocess
logger.info(f"Torch Cuda is available: {torch.cuda.is_available()}")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
logger.info(f"Torch device: {device}")

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import FastaiLRFinder

from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - self.seq_len

    def __getitem__(self, index):
        return self.X[index:index+self.seq_len], self.y[index+self.seq_len]


class TSModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        n_features = kwargs.get("n_features")
        
        if n_features is None:
            raise Exception("n_features expected as parameter")
        
        n_hidden = kwargs.get("n_hidden")
        if n_hidden is None:
            n_hidden = 64

        n_layers = kwargs.get("n_layers")
        if n_layers is None:
            n_layers = 1

        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.0
        )
        self.linear = nn.Linear(n_hidden, 1)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        lstm_out = hidden[-1]  # output last hidden state output
        y_pred = self.linear(lstm_out)
        
        return y_pred.squeeze()

class MLP(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

    seq_len = kwargs.get("seq_len")
    if seq_len is None:
        raise Exception("seq_len expected as parameter")
    
    n_features = kwargs.get("n_features")
    if n_features is None:
        raise Exception("n_features expected as parameter")

    batch_size = kwargs.get("batch_size")
    if batch_size is None:
        raise Exception("batch_size expected as parameter")
    
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(n_features*seq_len, batch_size),
      nn.ReLU(),
      nn.Linear(batch_size, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )

  def forward(self, x):
    return self.layers(x)

available_models = {
    "LSTM" : TSModel,
    "MLP" : MLP
}

def train_model(
        train_df,
        test_df,
        label_name,
        sequence_length,
        batch_size,
        n_epochs,
        n_epochs_stop,
        learning_rate,
        min_delta_stop,
        base_path,
        model_name,
        model_id
):
    """Train model."""
    logger.info("Starting with model training...")

    
    train_df = train_df.astype(np.float32)
    test_df = test_df.astype(np.float32)

    # create dataloaders
    train_dataset = TimeSeriesDataset(np.array(train_df), np.array(train_df[label_name]), seq_len=sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TimeSeriesDataset(np.array(test_df), np.array(test_df[label_name]), seq_len=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # set up training
    n_features = train_df.shape[1]
    model_class = available_models.get(model_name)
    logger.info(f"USING MODEL CLASS {model_class}")
    if model_class is None:
        raise Exception(f"Unrecognized model_name: {model_name}. \nAvailable models: {list(available_models.keys())}")
    model = model_class(
        n_features=n_features,
        seq_len=sequence_length,
        batch_size=batch_size
    ).to(device)

    criterion = torch.nn.MSELoss()  # L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    lr_finder = FastaiLRFinder()
    to_save = {"model": model, "optimizer": optimizer}

    try:
        with lr_finder.attach(trainer, to_save=to_save, start_lr=10e-5, end_lr=10e-1, diverge_th=5, num_iter=100) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(train_loader)

        logger.info("Finding best learning rate.")
        # Plot lr_finder results (requires matplotlib)
        ax = lr_finder.plot(skip_start=0, skip_end=0)
        ax.figure.savefig(Path(base_path, f'model_{model_id}_LR_plot.jpg'))

        # get lr_finder suggestion for lr
        logger.info(f"Suggested LR from ignite: {lr_finder.lr_suggestion()}")
        lr_finder.apply_suggested_lr(optimizer)
    except Exception as e:
        logger.warning(f"Could not get suggested lr: {e}")

    train_hist = []
    test_hist = []

    # start training
    best_loss = np.inf
    epochs_no_improve = 0
    logger.info("Beginning epochs")
    for epoch in range(1, n_epochs+1):
        running_loss = 0
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            data = torch.Tensor(np.array(data)).to(device)
            output = model(data)
            loss = criterion(output.flatten(), target.type_as(output))
            # if type(criterion) == torch.nn.modules.loss.MSELoss:
            #     loss = torch.sqrt(loss)  # RMSE
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss /= len(train_loader)
        train_hist.append(running_loss)

        # test loss
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = torch.Tensor(np.array(data)).to(device)
                output = model(data)
                loss = criterion(output.flatten(), target.type_as(output))
                test_loss += loss.item()
            test_loss /= len(test_loader)
            test_hist.append(test_loss)

            # early stopping
            if test_loss < best_loss + min_delta_stop:
                best_loss = test_loss
                torch.save(model.state_dict(), Path(base_path, f'model_{model_id}.pt'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                logger.info("Early stopping.")
                break

        logger.debug(f'Epoch {epoch} train loss: {round(running_loss,8)} test loss: {round(test_loss,8)}')

        hist = pd.DataFrame()
        hist['training_loss'] = train_hist
        hist['test_loss'] = test_hist

    logger.info("Completed.")

    return hist