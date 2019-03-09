"""
Defines the environment for portfolio management
"""

from collections import OrderedDict
from pprint import pprint

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchlib.common import eps

date_format = '%Y-%m-%d'


def sharpe(returns, freq=30, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.idxmax():].min()
    return (trough - peak) / (peak + eps)


class StockObservation(object):
    """
    A stock observation consists of previous day numeric, current day numeric and current day news
    """

    def __init__(self, numeric, news, prev_numeric, current_date):
        self.prev_numeric = prev_numeric
        self.numeric = numeric
        self.news = news
        self.current_date = current_date

    def get_close_ratio(self):
        return self.numeric[:, -1] / self.prev_numeric[:, -1]

    def get_news(self):
        return self.news

    def get_current_date(self):
        return self.current_date


class DataGenerator(object):
    def __init__(self, pd_frame: OrderedDict, total_steps=260, start_date=None):
        """

        Args:
            pd_frame: a dictionary of panda frame that provide stock price info and news or tweets.
                      The key is the stock name.
            total_steps: total number of steps to simulate
            start_date: date to start simulation. Used to compare algorithms.
        """
        self.pd_frame = pd_frame
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.total_steps = total_steps
        any_stock_frame = pd_frame.get(list(pd_frame.keys())[0])
        self.date_index = any_stock_frame.index
        self.earliest_record_date = pd.Timestamp(any_stock_frame.index.values[0])
        self.last_record_date = pd.Timestamp(any_stock_frame.index.values[-1])
        self.news_title = ['Top{}'.format(i + 1) for i in range(25)]

        if self.start_date and self.start_date not in self.date_index:
            raise ValueError('Start date not in date index. Must from {} to {} and must be weekdays'.format(
                self.earliest_record_date.strftime(date_format), self.last_record_date.strftime(date_format)))

    def _get_data_by_date(self, date):
        prices = []
        news = []
        for stock in self.pd_frame:
            current_frame = self.pd_frame[stock]
            line = current_frame.loc[date]
            open = line['Open']
            high = line['High']
            low = line['Low']
            close = line['Close']
            current_price = np.array([open, high, low, close])
            current_news = []
            for title in self.news_title:
                current_news.append(line[title])

            prices.append(current_price)
            news.append(current_news)

        prices = np.array(prices)
        return prices, news

    def _get_next_date(self):
        next_date = self.current_date + pd.Timedelta(days=1)
        while next_date not in self.date_index:
            if next_date > self.last_record_date:
                raise ValueError('Current date {} exceeds max date {}'.format(self.current_date,
                                                                              self.last_record_date))
            next_date += pd.Timedelta(days=1)
        return next_date

    def step(self):
        """

        Returns: a data dictionary map from stock name to (open, high, low, close, Top1, ..., Top25)

        """
        self.steps += 1
        self.current_date = self._get_next_date()
        if self.current_date == self.last_record_date or self.steps == self.total_steps:
            done = True
        else:
            done = False

        current_prices, current_news = self._get_data_by_date(self.current_date)
        obs = StockObservation(current_prices, current_news, self.prev_prices, self.current_date)
        self.prev_prices = current_prices.copy()
        return obs, done

    def reset(self):
        self.steps = 0
        if self.start_date:
            self.current_date = self.start_date
        else:
            idx = np.random.randint(low=0, high=self.date_index.size - self.total_steps - 1)
            self.current_date = self.date_index[idx]  # random select

        prev_prices, _ = self._get_data_by_date(self.current_date)
        self.current_date = self._get_next_date()
        current_prices, current_news = self._get_data_by_date(self.current_date)
        obs = StockObservation(current_prices, current_news, prev_prices, self.current_date)
        self.prev_prices = current_prices.copy()
        return obs


class PortfolioSim(object):
    def __init__(self, asset_names=(), trading_cost=0.0025, time_cost=0.0, total_steps=260):
        self.asset_names = asset_names
        self.trading_cost = trading_cost
        self.time_cost = time_cost
        self.total_steps = total_steps

    def step(self, w1, y1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        y1 - price relative vector also called return in this timestamp
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape.'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        w0 = self.w0
        p0 = self.p0

        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into

        mu1 = self.trading_cost * (np.abs(dw1 - w0)).sum()  # (eq16) cost to change portfolio

        assert mu1 < 1.0, 'Cost is larger than current holding'

        p1 = p0 * (1 - mu1) * np.dot(y1, w0)  # (eq11) final portfolio value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (p0 + eps))  # log rate of return
        reward = r1  # (22) average logarithmic accumulated return
        # remember for next step
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = p1 == 0

        self.w0 = w1  # remember the action for calculating reward at next timestamp

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.p0 = 1.0
        self.w0 = np.array([1.0] + [0.0] * len(self.asset_names))


class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,
                 pd_frame_dict,
                 total_steps=260,  # 1 years
                 trading_cost=0.0025,
                 time_cost=0.00,
                 start_date=None
                 ):
        """
        An environment for financial portfolio management.
        Params:
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
            start_idx - The number of days from '2012-08-13' of the dataset
            sample_start_date - The start date sampling from the history
        """
        self.num_stocks = len(pd_frame_dict.keys())

        self.src = DataGenerator(pd_frame_dict, total_steps=total_steps, start_date=start_date)

        self.sim = PortfolioSim(
            asset_names=pd_frame_dict.keys(),
            trading_cost=trading_cost,
            time_cost=time_cost,
            total_steps=total_steps)

        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=(self.num_stocks + 1,), dtype=np.float32)  # include cash

    def step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        if isinstance(action, list):
            action = np.array(action)

        assert isinstance(action, np.ndarray), 'Action must be a numpy array'

        np.testing.assert_almost_equal(
            action.shape,
            (len(self.sim.asset_names) + 1,)
        )

        # normalise just in case
        action = np.clip(action, 0, 1)

        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)  # so if weights are all zeros we normalise to [1,0...]

        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        observation, done1 = self.src.step()

        # relative price vector of last observation day (close/close)
        y1 = observation.get_close_ratio()
        y1 = np.insert(y1, 0, 1.)
        reward, info, done2 = self.sim.step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = self.current_date
        self.current_date = observation.get_current_date()

        self.infos.append(info)

        return observation, reward, done1 or done2, info

    def reset(self):
        self.infos = []
        self.sim.reset()
        observation = self.src.reset()
        self.current_date = observation.get_current_date()
        return observation

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            self.plot()

    def seed(self, seed=None):
        self.seed = seed

    def plot(self):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos)
        df_info.set_index('date', inplace=True)
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
        df_info[["portfolio_value", "market_value"]].plot(title=title, fig=plt.gcf(), rot=30)


class PortfolioEnvPriceOnly(PortfolioEnv):
    def __init__(self,
                 pd_frame_dict,
                 total_steps=260,  # 1 years
                 trading_cost=0.0025,
                 time_cost=0.00,
                 start_date=None
                 ):
        super(PortfolioEnvPriceOnly, self).__init__(pd_frame_dict, total_steps,
                                                    trading_cost, time_cost, start_date)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

    def step(self, action):
        observation, reward, done, info = super(PortfolioEnvPriceOnly, self).step(action)
        close_ratio = self.get_normalized_close_ratio(observation)
        return close_ratio, reward, done, info

    def reset(self):
        observation = super(PortfolioEnvPriceOnly, self).reset()
        return self.get_normalized_close_ratio(observation)

    def get_normalized_close_ratio(self, observation):
        close_ratio = observation.get_close_ratio()
        close_ratio = (close_ratio - 1.) * 100
        return close_ratio


class PortfolioEnvPriceOnlyRewardShape(PortfolioEnvPriceOnly):
    def step(self, action):
        close_ratio, reward, done, info = super(PortfolioEnvPriceOnlyRewardShape, self).step(action)
        if reward > 0:
            reward = 1
        else:
            reward = -1

        return close_ratio, reward, done, info
