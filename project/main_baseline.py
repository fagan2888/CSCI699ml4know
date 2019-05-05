import pprint
from collections import OrderedDict

import numpy as np
import torch.optim
import torchlib.deep_rl as deep_rl
import torchlib.deep_rl.policy_gradient.ppo as ppo

from portfolio_mangement.agents import NewsOnlyPolicyModule, PriceOnlyPolicyModule
from portfolio_mangement.envs import PortfolioEnv, PortfolioEnvNewsOnlyWrapper, PortfolioRewardWrapper, \
    PortfolioEnvPriceOnlyWrapper
from portfolio_mangement.utils.data import read_djia_observation


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--gae_lambda', type=float, default=0.98)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.00)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--recurrent', '-re', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--ep_len', '-ep', type=int, default=100)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--observation', type=str, default='price', choices=['price', 'news'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    return parser


def build_agent(args, env):
    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    recurrent = args.recurrent
    hidden_size = args.hidden_size

    if args.observation == 'price':
        policy_net = PriceOnlyPolicyModule(args.nn_size, ob_dim, ac_dim, recurrent, hidden_size)
    elif args.observation == 'news':
        policy_net = NewsOnlyPolicyModule(num_stocks=ac_dim - 1, seq_length=128, recurrent=args.recurrent,
                                          hidden_size=args.hidden_size, freeze_embedding=True)
    else:
        raise ValueError('Unknown observation type {}'.format(args.observation))

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), args.learning_rate)

    gae_lambda = args.gae_lambda

    if recurrent:
        init_hidden_unit = np.zeros(shape=(hidden_size))
    else:
        init_hidden_unit = None

    agent = ppo.Agent(policy_net, policy_optimizer,
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

    checkpoint_path = 'checkpoint/{}_only_ppo.ckpt'.format(args.observation)

    if args.observation == 'price':
        obs_wrapper = PortfolioEnvPriceOnlyWrapper
    elif args.observation == 'news':
        obs_wrapper = PortfolioEnvNewsOnlyWrapper
    else:
        raise ValueError('Unknown observation type {}'.format(args.observation))

    if not test:
        pd_frame_dict['DJIA'] = train_pd_frame
        env = PortfolioEnv(pd_frame_dict, total_steps=max_path_length)
        env = obs_wrapper(env)
        env = PortfolioRewardWrapper(env)

        agent = build_agent(args, env)
        ppo.train(args.exp_name, env, agent, args.n_iter, args.discount, args.batch_size, max_path_length,
                  logdir='runs/{}_only_ppo'.format(args.observation), seed=args.seed,
                  checkpoint_path=checkpoint_path)

    else:
        pd_frame_dict['DJIA'] = test_pd_frame
        print('Test on Reward Shaping Env')
        env = PortfolioEnv(pd_frame_dict, total_steps=max_path_length)
        env = obs_wrapper(env)
        reward_shape_env = PortfolioRewardWrapper(env)
        agent = build_agent(args, reward_shape_env)
        agent.load_checkpoint(checkpoint_path)
        deep_rl.test(reward_shape_env, agent, args.n_iter, seed=args.seed)

        print('Test on Original Env')
        deep_rl.test(env, agent, args.n_iter, seed=args.seed)
        env.render()
