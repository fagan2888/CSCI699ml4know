"""
Train classifier to predict class labels
"""

from collections import OrderedDict

import numpy as np
import torch.nn as nn
import torch.optim
from torchlib.dataset.utils import create_data_loader
from torchlib.trainer import BinaryClassifier

from portfolio_mangement.agents import NewsPredictorModule
from portfolio_mangement.envs import PortfolioEnv, PortfolioEnvObsReshapeWrapper
from portfolio_mangement.utils.data import read_djia_observation


def create_dataset(pd_frame, max_seq_length):
    pd_frame_dict = OrderedDict()
    pd_frame_dict['DJIA'] = pd_frame
    env = PortfolioEnv(pd_frame_dict, total_steps=-1)
    env = PortfolioEnvObsReshapeWrapper(env, max_seq_length=max_seq_length)
    obs = env.reset()
    labels = []
    observation = []
    done = False
    while not done:
        close_ratio, news = obs
        observation.append(news)
        labels.append(close_ratio)
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
    observation = observation[:-1]
    labels = labels[1:]
    observation = np.stack(observation, axis=0)
    labels = (np.stack(labels, axis=0) > 0).astype(np.float)
    return observation, labels


if __name__ == '__main__':
    pd_frame = read_djia_observation('data/stocknews')
    train_pd_frame, test_pd_frame = np.split(pd_frame, [int(0.8 * len(pd_frame))])

    max_seq_length = 64

    print('Collecting dataset')
    train_obs, train_labels = create_dataset(train_pd_frame, max_seq_length)
    test_obs, test_labels = create_dataset(test_pd_frame, max_seq_length)
    print('Finish')

    model = NewsPredictorModule(seq_length=max_seq_length, freeze_embedding=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    classifier = BinaryClassifier(model, optimizer, nn.BCELoss())

    train_loader = create_data_loader((train_obs, train_labels), batch_size=32, shuffle=True, drop_last=False)
    val_loader = create_data_loader((test_obs, test_labels), batch_size=32, shuffle=False, drop_last=False)

    checkpoint_path = 'checkpoint/baseline_predictor.ckpt'

    classifier.train(epoch=100, train_data_loader=train_loader, val_data_loader=val_loader,
                     checkpoint_path=checkpoint_path)
