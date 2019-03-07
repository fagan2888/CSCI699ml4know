"""
Data utilities for import various data source and formatting
"""

import os

import pandas as pd


def read_djia_observation(folder_path):
    """ Read Dow Johns Index Average as observation in RL environment

    Args:
        folder_path: path to stocknews folder containing three csv files.

    Returns: panda data frame containing (date, open, high, low, close, volume, top1, top2, ..., top25)

    """
    combined_csv_path = os.path.join(folder_path, 'Combined_News_DJIA.csv')
    dija_table_csv = os.path.join(folder_path, 'DJIA_table.csv')
    news_table = pd.read_csv(combined_csv_path)
    djia_table = pd.read_csv(dija_table_csv)
    assert news_table.shape[0] == djia_table.shape[0]
    news_table = news_table.set_index(pd.DatetimeIndex(news_table['Date'])).drop('Date', axis=1)
    djia_table = djia_table.set_index(pd.DatetimeIndex(djia_table['Date'])).drop('Date', axis=1)
    djia_pd_frame = news_table.join(djia_table)
    djia_pd_frame = djia_pd_frame.sort_index()
    assert djia_pd_frame.shape[0] == djia_table.shape[0]
    return djia_pd_frame


def read_djia_sentiment(folder_path):
    """ Read DJIA for training sentiment model. Note the label is calculated using Close price

    Args:
        folder_path: path to stocknews folder containing three csv files.

    Returns:
        sentences: a list of list of 25 sentences.
        labels: A list of labels

    """
    return [[]], []


if __name__ == '__main__':
    djia_pd_frame = read_djia_observation('../../data/stocknews')
