import pprint
from collections import OrderedDict

import numpy as np
import torch.optim
import torchlib.deep_rl as deep_rl
import torchlib.deep_rl.policy_gradient.ppo as ppo
import torch.nn as nn

from torchlib.common import FloatTensor

from portfolio_mangement.envs import PortfolioEnvPriceOnlyRewardShape, PortfolioEnvPriceOnly
from portfolio_mangement.utils.data import read_djia_observation


class PolicyContinuous(nn.Module):
    def __init__(self, nn_size, state_dim, action_dim, recurrent=False, hidden_size=20):
        super(PolicyContinuous, self).__init__()
        random_number = np.random.randn(action_dim, action_dim)
        random_number = np.dot(random_number.T, random_number)

        self.logstd = torch.nn.Parameter(torch.tensor(random_number, requires_grad=True).type(FloatTensor))
        self.model = nn.Sequential(
            nn.Linear(state_dim, nn_size),
            nn.ReLU(),
            nn.Linear(nn_size, nn_size),
            nn.ReLU(),
        )

        if recurrent:
            linear_size = hidden_size
        else:
            linear_size = nn_size

        self.action_head = nn.Sequential(
            nn.Linear(linear_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(linear_size, 1)

        self.recurrent = recurrent

        if self.recurrent:
            self.gru = nn.GRU(nn_size, hidden_size)

    def forward(self, state, hidden):
        x = self.model.forward(state)
        if self.recurrent:
            x, hidden = self.gru.forward(x.unsqueeze(0), hidden.unsqueeze(0))
            x = x.squeeze(0)
            hidden = hidden.squeeze(0)
        mean = self.action_head.forward(x)
        value = self.value_head.forward(x)
        return (mean, self.logstd), hidden, value.squeeze(-1)

def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--gae_lambda', type=float, default=0.98)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--recurrent', '-re', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=20)
    parser.add_argument('--ep_len', '-ep', type=int, default=280)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    return parser


def build_agent(args, env):
    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    recurrent = args.recurrent
    hidden_size = args.hidden_size

    policy_net = PolicyContinuous(args.nn_size, ob_dim, ac_dim, recurrent, hidden_size)

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), args.learning_rate)

    gae_lambda = args.gae_lambda

    if recurrent:
        init_hidden_unit = np.zeros(shape=(hidden_size))
    else:
        init_hidden_unit = None

    agent = ppo.Agent(policy_net, policy_optimizer,
                      discrete=False,
                      init_hidden_unit=init_hidden_unit,
                      lam=gae_lambda,
                      clip_param=args.clip_param,
                      entropy_coef=args.entropy_coef, value_coef=args.value_coef)

    return agent


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    pprint.pprint(vars(args))

    max_path_length = args.ep_len if args.ep_len > 0 else None

    test = args.test

    pd_frame = read_djia_observation('data/stocknews')
    pd_frame_dict = OrderedDict()

    train_pd_frame, test_pd_frame = np.split(pd_frame, [int(0.8 * len(pd_frame))])
    checkpoint_path = 'checkpoint/price_only_ppo.ckpt'

    if not test:
        pd_frame_dict['DJIA'] = train_pd_frame
        env = PortfolioEnvPriceOnlyRewardShape(pd_frame_dict, total_steps=max_path_length)

        agent = build_agent(args, env)
        ppo.train(args.exp_name, env, agent, args.n_iter, args.discount, args.batch_size, max_path_length,
                  logdir='runs/price_only_ppo', seed=args.seed, checkpoint_path=checkpoint_path)

    else:
        pd_frame_dict['DJIA'] = test_pd_frame
        print('Test on Reward Shaping Env')
        env = PortfolioEnvPriceOnlyRewardShape(pd_frame_dict, total_steps=max_path_length)
        agent = build_agent(args, env)
        agent.load_checkpoint(checkpoint_path)
        deep_rl.test(env, agent, args.n_iter, seed=args.seed)

        print('Test on Original Env')
        env = PortfolioEnvPriceOnly(pd_frame_dict, total_steps=max_path_length)
        deep_rl.test(env, agent, args.n_iter, seed=args.seed)
