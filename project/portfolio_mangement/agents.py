import numpy as np
import torch
import torch.nn as nn
from torchlib.common import FloatTensor
from pytorch_pretrained_bert import BertModel

from torchlib.utils.torch_layer_utils import conv2d_bn_relu_block

class PriceOnlyPolicyModule(nn.Module):
    def __init__(self, nn_size, state_dim, action_dim, recurrent=False, hidden_size=20):
        super(PriceOnlyPolicyModule, self).__init__()
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


class NewsOnlyPolicyModule(nn.Module):
    """
    A predictor for news data. The input is indices of 25
    """
    def __init__(self, num_stocks, num_news, seq_length, recurrent):
        super(NewsOnlyPolicyModule, self).__init__()
        self.embedding = BertModel.from_pretrained('bert-base-uncased').embeddings
        self.cnn = nn.Sequential(
            *conv2d_bn_relu_block(in_channels=768, )
        )



    def forward(self, news):
        """

        Args:
            news: shape is (batch_size, num_stocks, num_news, seq_length) with idx of words.

        Returns: the probability of stock goes up (batch_size, num_stocks)

        """

