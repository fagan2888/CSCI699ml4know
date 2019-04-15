"""
Data utilities for import various data source and formatting
"""

import os

import numpy as np
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


def get_djia_dual_attention_path(window_length, sentiment_analyzer, regression, train_val_test_split):
    return "{}_{}_{}_{}.npz".format(window_length, sentiment_analyzer, regression, train_val_test_split)


def create_djia_dual_attention_dataset(folder_path, window_length, sentiment_analyzer, regression=False,
                                       train_val_test_split=(0.8, 0.1, 0.1), out_data_path=None):
    """ Here the time series should be close ratio (close_ratio - 1.) * 100.

    Args:
        folder_path: path to stocknews folder containing three csv files.
        window_length: window length
        sentiment_analyzer: sentiment analyzer that can transform a sentence into a single polarity.
        regression: if regression, output will be numbers. Otherwise, output will be labels.

    Returns: train, val, test data loader, each containing:
        (y: (num, window_length), news_sentiment: (num, 25, window_length), label: (num,))

    """
    if not out_data_path:
        out_data_path = get_djia_dual_attention_path(window_length, sentiment_analyzer, regression,
                                                     train_val_test_split)

    def create_dataset(pd_frame):
        close_price = pd_frame['Close'].values
        close_ratio = close_price[1:] / close_price[:-1]
        close_ratio = (close_ratio - 1.) * 100  # (1988,)

        all_news_sentiment = []
        for news_index in range(25):
            news = pd_frame['Top{}'.format(news_index + 1)].values[1:]  # ignore the first one.
            current_news_sentiment = []
            for n in news:
                current_news_sentiment.append(sentiment_analyzer.predict(n))
            current_news_sentiment = np.array(current_news_sentiment)
            all_news_sentiment.append(current_news_sentiment)

        all_news_sentiment = np.array(all_news_sentiment)  # (25, 1988)

        history = []
        news_sentiment = []
        label = []
        for i in range(close_ratio.shape[0] - window_length):
            history.append(close_ratio[i:i + window_length])
            label.append(close_ratio[i + window_length])
            news_sentiment.append(all_news_sentiment[:, i:i + window_length])
        history = np.array(history)
        news_sentiment = np.array(news_sentiment)
        label = np.array(label)

        if not regression:
            label = (label > 0).astype(np.int)

        return history, news_sentiment, label

    djia_pd_frame = read_djia_observation(folder_path)


if __name__ == '__main__':
    djia_pd_frame = read_djia_observation('../../data/stocknews')
