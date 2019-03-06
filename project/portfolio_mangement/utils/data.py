"""
Data utilities for import various data source and formatting
"""

import pandas as pd


def read_djia_observation(folder_path):
    """ Read Dow Johns Index Average as observation in RL environment

    Args:
        folder_path: path to stocknews folder containing three csv files.

    Returns: panda data frame containing (date, open, high, low, close, volume, top1, top2, ..., top25)

    """
    return pd.DataFrame()



def read_djia_sentiment(folder_path):
    """ Read DJIA for training sentiment model

    Args:
        folder_path: path to stocknews folder containing three csv files.

    Returns:
        sentences: a list of list of 25 sentences.
        labels: A list of labels

    """
    return [[]], []

