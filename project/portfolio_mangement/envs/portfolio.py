"""
Defines the environment for portfolio management
"""

import pandas as pd
import gym
import numpy as np
from typing import Dict
from pprint import pprint


from torchlib.common import eps

class DataGenerator(object):
    def __init__(self, pd_frame: Dict, total_steps=260, start_date=None):
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


    def step(self):
        """

        Returns: a data dictionary map from stock name to (open, high, low, close, Top1, ..., Top25)

        """
        current_data = {}
        for stock in self.pd_frame:
            current_frame = self.pd_frame[stock]
            line = current_frame.loc[self.current_date]
            open = line['Open']
            high = line['High']
            low = line['Low']
            close = line['Close']
            price = [open, high, low, close]
            news = []
            for title in self.news_title:
                news.append(line[title])

            data = (price, news)
            current_data[stock] = data

        previous_date = self.current_date  # datetime is immutable

        self.current_date += pd.Timedelta(days=1)
        while self.current_date not in self.date_index:
            if self.current_date > self.last_record_date:
                raise ValueError('Current date {} exceeds max date {}'.format(self.current_date,
                                                                              self.last_record_date))
            self.current_date += pd.Timedelta(days=1)

        return current_data, previous_date

    def reset(self):
        if self.start_date:
            self.current_date = self.start_date
        else:
            idx = np.random.randint(low=0, high=self.date_index.length - self.total_steps)
            self.current_date = self.date_index[idx]   # random select



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
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
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
        reward = r1 / self.total_steps  # (22) average logarithmic accumulated return
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
                 history,
                 abbreviation,
                 steps=730,  # 2 years
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 start_idx=0,
                 sample_start_date=None
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
        self.window_length = window_length
        self.num_stocks = history.shape[0]
        self.start_idx = start_idx

        self.src = DataGenerator(history, abbreviation, steps=steps, window_length=window_length, start_idx=start_idx,
                                 start_date=sample_start_date)

        self.sim = PortfolioSim(
            asset_names=abbreviation,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)

        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(self.src.asset_names) + 1,), dtype=np.float32)  # include cash

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(abbreviation), window_length,
                                                                                 history.shape[-1]), dtype=np.float32)

    def step(self, action):
        return self._step(action)

    def _step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
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

        observation, done1, ground_truth_obs = self.src._step()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # relative price vector of last observation day (close/open)
        close_price_vector = observation[:, -1, 3]
        open_price_vector = observation[:, -1, 0]
        y1 = close_price_vector / open_price_vector
        reward, info, done2 = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step)
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs

        self.infos.append(info)

        return observation, reward, done1 or done2, info

    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        observation, ground_truth_obs = self.src.reset()
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info = {}
        info['next_obs'] = ground_truth_obs
        return observation, info

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            self.plot()

    def render(self, mode='human', close=False):
        return self._render(mode='human', close=False)

    def plot(self):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos)
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
        df_info[["portfolio_value", "market_value"]].plot(title=title, fig=plt.gcf(), rot=30)
