"""
Train classifier or regressor using dual attention network
"""


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_hidden_size', type=int, default=32)
    parser.add_argument('--temporal_hidden_size', type=int, default=32)
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--regression', action='store_true')  # by default, it's classification

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    return parser


def get_checkpoint_path(input_hidden_size, temporal_hidden_size, window_size, regression,
                        sentiment_analyzer):
    return 'checkpoint/{}_{}_{}_{}_{}'.format(input_hidden_size, temporal_hidden_size, window_size,
                                              regression, sentiment_analyzer)


if __name__ == '__main__':
    parser = make_parser()
    args = vars(parser.parse_args())

    import pprint

    import torch.nn as nn
    import torch.optim
    from torchlib.trainer import Trainer
    from torchlib.dataset.utils import create_data_loader

    from portfolio_mangement.models import DualAttentionRNN
    from portfolio_mangement.utils.data import load_djia_dual_attention_dataset
    from portfolio_mangement.sentiment_analyzer import NLTKSentimentAnalyzer

    pprint.pprint(args)

    input_hidden_size = args['input_hidden_size']
    temporal_hidden_size = args['temporal_hidden_size']
    window_size = args['window_size']
    regression = args['regression']

    # define model
    net = DualAttentionRNN(input_hidden_size=input_hidden_size,
                           temporal_hidden_size=temporal_hidden_size,
                           num_driving_series=25,
                           window_size=window_size,
                           regression=regression)

    # define trainer
    learning_rate = args['learning_rate']
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    if regression:
        criterion = nn.MSELoss()
        metrics = None
    else:
        criterion = nn.CrossEntropyLoss()
        metrics = 'accuracy'

    scheduler = None
    trainer = Trainer(model=net, optimizer=optimizer, loss=criterion, metrics=metrics, scheduler=scheduler)

    # define dataset
    sentiment_analyzer = NLTKSentimentAnalyzer()
    out = load_djia_dual_attention_dataset(window_length=window_size, sentiment_analyzer=sentiment_analyzer,
                                           regression=regression)
    train_history, train_news, train_label = out[0]
    val_history, val_news, val_label = out[1]
    test_history, test_news, test_label = out[2]

    batch_size = args['batch_size']
    epoch = args['epoch']

    train_loader = create_data_loader(out[0], batch_size=batch_size)
    val_loader = create_data_loader(out[1], batch_size=batch_size)
    test_loader = create_data_loader(out[2], batch_size=batch_size)

    checkpoint_path = get_checkpoint_path(input_hidden_size, temporal_hidden_size, window_size,
                                          regression, sentiment_analyzer)

    trainer.fit(train_data_loader=train_loader, num_inputs=2, epochs=epoch, val_data_loader=val_loader,
                checkpoint_path=checkpoint_path)
