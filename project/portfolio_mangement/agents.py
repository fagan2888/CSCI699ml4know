import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from torchlib.common import FloatTensor, LongTensor
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


class NewsPredictorModule(nn.Module):
    """
    A predictor for news data. The input is indices of 25
    """

    def __init__(self, seq_length):
        """

        Args:
            num_stocks: defines the output dimension
            num_news: number of stocks
            seq_length: sequence length
            recurrent: whether use recurrent unit
        """
        super(NewsPredictorModule, self).__init__()
        self.embedding = BertModel.from_pretrained('bert-base-uncased').embeddings
        self.model = nn.Sequential(
            *conv2d_bn_relu_block(in_channels=768, out_channels=64, kernel_size=(1, 8), stride=(1, 4),
                                  padding=(0, 2), normalize=False),
            *conv2d_bn_relu_block(in_channels=64, out_channels=32, kernel_size=(1, 4), stride=(1, 2),
                                  padding=(0, 1), normalize=False),
            *conv2d_bn_relu_block(in_channels=32, out_channels=16, kernel_size=(1, 4), stride=(1, 2),
                                  padding=(0, 1), normalize=False),
        )

        linear_size = 16 * seq_length // 16

        self.linear = nn.Sequential(
            nn.Linear(linear_size, 1),
            nn.Sigmoid()
        )

    def forward(self, state):
        """

        Args:
            state: shape is (batch_size, num_stocks, num_news, seq_length) with idx of words.

        Returns: the probability of stock goes up (batch_size, num_stocks)

        """
        state = state.type(LongTensor)
        batch_size, num_stocks, num_news, seq_length = state.shape
        state = state.view(batch_size * num_stocks, num_news, seq_length)
        state = state.view(batch_size * num_stocks, num_news * seq_length)
        embedding = self.embedding.forward(state)
        embedding = embedding.view(batch_size * num_stocks, num_news, num_stocks)
        cnn_out = self.model.forward(embedding)  # (batch*num_stock, num_news, 8, 16)
        # average with all the news
        cnn_out = torch.mean(cnn_out, dim=1)  # (batch*num_stock, 8, 16)
        x = cnn_out.view(cnn_out.size(0), -1)
        prob = self.linear.forward(x)
        prob = prob.view(batch_size, num_stocks)
        return prob


class NewsOnlyPolicyModule(nn.Module):
    def __init__(self, num_stocks, seq_length, recurrent=False, hidden_size=20):
        super(NewsOnlyPolicyModule, self).__init__()
        self.model = NewsPredictorModule(seq_length)

        if recurrent:
            linear_size = hidden_size
        else:
            linear_size = num_stocks

        action_dim = num_stocks + 1

        self.action_head = nn.Sequential(
            nn.Linear(linear_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(linear_size, 1)

        self.recurrent = recurrent

        if self.recurrent:
            self.gru = nn.GRU(linear_size, hidden_size)

    def forward(self, state, hidden):
        x = self.model.forward(state)
        if self.recurrent:
            x, hidden = self.gru.forward(x.unsqueeze(0), hidden.unsqueeze(0))
            x = x.squeeze(0)
            hidden = hidden.squeeze(0)
        mean = self.action_head.forward(x)
        value = self.value_head.forward(x)
        return (mean, self.logstd), hidden, value.squeeze(-1)
