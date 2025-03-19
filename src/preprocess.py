""" Module for data preparation. """


import yaml
import joblib
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import logging
logger = logging.getLogger("experiment_worker")

def clean_data(
        df,
):
    """Sort by date and fill NA values."""
    # sort by date
    df_clean = df.sort_values(by='Date').reset_index(drop=True)
    # drop NaN
    df_clean = df_clean.fillna(0)

    return df_clean


def create_features(
        df
):
    """Creates new features."""
    

    
    # add date-derived features
    df['Day_Of_Week'] = np.sin(pd.DatetimeIndex(df['Date']).dayofweek * 2 * np.pi/7.)
    df['Month_Of_Year'] = np.sin(pd.DatetimeIndex(df['Date']).month * 2 * np.pi/12.)
    df['Quarter_Of_Year'] = np.sin(pd.DatetimeIndex(df['Date']).quarter * 2 * np.pi/4.)

    # add intraday gaps
    df['High_Low_Pct'] = (df.High - df.Low) / df.Low  # percentage intraday change
    df['Open_Close_Pct'] = (df.Open.shift(-1) - df.Close) / df.Close  # percentage change using next day open
    
    # drop rows with missing values
    df = df.fillna(0)
    
    return df


def split_data(
        df,
        train_frac
):
    train_size = int(len(df) * train_frac)
    train_df, test_df = df[:train_size], df[train_size:]

    return train_df, test_df, train_size


def rescale_data(
        df,
        base_path
):
    """Rescale all features using MinMaxScaler() to same scale, between 0 and 1."""
    
    scaler = MinMaxScaler()
    scaler = scaler.fit(df)

    df_scaled = pd.DataFrame(
        scaler.transform(df),
        index=df.index,
        columns=df.columns)

    # save trained data scaler
    joblib.dump(scaler, Path(base_path, 'scaler.gz'))
    
    return df_scaled


def prep_data_basic(
        df,
        train_frac,
        base_path,
        keep_date=False,
        scale_features=True,
        add_noise=False,
        feature_to_base_noise=None,
):
    logger.info("Starting with data preparation...")

    df_clean = clean_data(df)

    # split into train/test datasets
    train_df, test_df, train_size = split_data(df_clean, train_frac)

    train_date_series = train_df["Date"].copy()
    test_date_series = test_df["Date"].copy()

    columns_of_interest = [column for column in df.columns if column != "Date"]
    
    train_df = train_df[columns_of_interest]
    test_df = test_df[columns_of_interest]

    if scale_features:
        # rescale data only in training for not having influence from test data
        train_scaled_df = rescale_data(train_df, base_path)

        scaler = joblib.load(Path(base_path, 'scaler.gz'))
        test_scaled_df = pd.DataFrame(
            scaler.transform(test_df),
            index=test_df.index,
            columns=test_df.columns
        )
    else:
        train_scaled_df = train_df
        test_scaled_df = test_df

    if keep_date:
        train_scaled_df["Date"] = train_date_series
        test_scaled_df["Date"] = test_date_series

    if add_noise:
        for df in [train_scaled_df, test_scaled_df]:
            if feature_to_base_noise is None:
                raise Exception("Feature to base noise is None, it should be provided.")
            df["Noise"] = pd.Series(
                np.random.normal(
                    train_scaled_df[feature_to_base_noise].mean(),
                    train_scaled_df[feature_to_base_noise].std(),
                    size=len(df)
                )
            ).set_axis(df.index)

    # save data
    train_scaled_df.to_csv(f'{base_path}/train.csv')
    test_scaled_df.to_csv(f'{base_path}/test.csv')
    logger.info("Completed.")

    return train_scaled_df, test_scaled_df