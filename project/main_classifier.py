"""
Train classifier to predict class labels
"""

import pprint
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


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--freeze_embedding', type=bool, default=False)
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--test', action='store_true')

    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)

    pd_frame = read_djia_observation('data/stocknews')
    train_pd_frame, test_pd_frame = np.split(pd_frame, [int(0.8 * len(pd_frame))])

    max_seq_length = args['max_seq_length']

    print('Collecting dataset')
    train_obs, train_labels = create_dataset(train_pd_frame, max_seq_length)
    test_obs, test_labels = create_dataset(test_pd_frame, max_seq_length)
    print('Finish')

    model = NewsPredictorModule(seq_length=max_seq_length, freeze_embedding=args['freeze_embedding'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    classifier = BinaryClassifier(model, optimizer, nn.BCELoss())

    train_loader = create_data_loader((train_obs, train_labels), batch_size=32, shuffle=True, drop_last=False)
    val_loader = create_data_loader((test_obs, test_labels), batch_size=32, shuffle=False, drop_last=False)

    checkpoint_path = 'checkpoint/baseline_predictor.ckpt'

    if args['test']:
        classifier.load_checkpoint(checkpoint_path, all=False)

        loss, acc = classifier.evaluation(train_loader)
        print('Loss {}, Acc {}'.format(loss, acc))

        loss, acc = classifier.evaluation(val_loader)
        print('Loss {}, Acc {}'.format(loss, acc))
    else:
        classifier.train(epoch=args['epoch'], train_data_loader=train_loader, val_data_loader=val_loader,
                         checkpoint_path=checkpoint_path)
