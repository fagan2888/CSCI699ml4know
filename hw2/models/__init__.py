from .PCNN import PCNN, PCNNTwoHead, PCNNRankLoss
from .RNN import RNNAttention, RNNAttentionRankLoss


def get_model(model_name):
    if model_name == 'PCNN':
        return PCNN
    elif model_name == 'PCNNTwoHead':
        return PCNNTwoHead
    elif model_name == 'PCNNRankLoss':
        return PCNNRankLoss
    elif model_name == 'RNNAttention':
        return RNNAttention
    elif model_name == 'RNNAttentionRankLoss':
        return RNNAttentionRankLoss
    else:
        raise NotImplementedError
