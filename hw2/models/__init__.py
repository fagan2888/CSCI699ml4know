from .PCNN import PCNN, PCNNTwoHead, PCNNRankLoss


def get_model(model_name):
    if model_name == 'PCNN':
        return PCNN
    elif model_name == 'PCNNTwoHead':
        return PCNNTwoHead
    elif model_name == 'PCNNRankLoss':
        return PCNNRankLoss
    else:
        raise NotImplementedError
