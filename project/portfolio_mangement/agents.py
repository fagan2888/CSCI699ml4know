import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from torchlib.common import FloatTensor, LongTensor
from torchlib.utils.torch_layer_utils import conv2d_bn_relu_block, freeze


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

    def __init__(self, seq_length, freeze_embedding=False):
        """

        Args:
            num_stocks: defines the output dimension
            num_news: number of stocks
            seq_length: sequence length
            recurrent: whether use recurrent unit
        """
        super(NewsPredictorModule, self).__init__()
        self.embedding = BertModel.from_pretrained('bert-base-uncased').embeddings
        if freeze_embedding:
            freeze(self.embedding)
        self.model = nn.Sequential(
            *conv2d_bn_relu_block(in_channels=768, out_channels=8, kernel_size=(1, 8), stride=(1, 4),
                                  padding=(0, 2), normalize=True),
            *conv2d_bn_relu_block(in_channels=8, out_channels=4, kernel_size=(1, 8), stride=(1, 4),
                                  padding=(0, 2), normalize=True),
        )

        linear_size = 4 * seq_length // 16

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
        state = state.view(batch_size * num_stocks * num_news, seq_length)
        embedding = self.embedding.forward(state)
        embedding = embedding.view(batch_size * num_stocks, num_news, seq_length, 768).permute(0, 3, 1, 2)
        cnn_out = self.model.forward(embedding)  # (batch*num_stock, num_news, 8, 16)
        # average with all the news
        cnn_out = torch.mean(cnn_out, dim=2)  # (batch, 16, num_stock, 8)
        x = cnn_out.view(cnn_out.size(0), -1)
        prob = self.linear.forward(x)
        prob = prob.view(batch_size, num_stocks)
        return prob


class NewsOnlyPolicyModule(nn.Module):
    def __init__(self, num_stocks, seq_length, recurrent=False, hidden_size=16, freeze_embedding=False):
        super(NewsOnlyPolicyModule, self).__init__()
        self.model = NewsPredictorModule(seq_length, freeze_embedding=freeze_embedding)

        action_dim = num_stocks + 1
        random_number = np.random.randn(action_dim, action_dim)
        random_number = np.dot(random_number.T, random_number)
        self.logstd = torch.nn.Parameter(torch.tensor(random_number, requires_grad=True).type(FloatTensor))

        if recurrent:
            linear_size = hidden_size
        else:
            linear_size = num_stocks

        self.action_head = nn.Sequential(
            nn.Linear(linear_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(linear_size, 1)

        self.recurrent = recurrent

        if self.recurrent:
            self.gru = nn.GRU(num_stocks, hidden_size)

    def forward(self, state, hidden):
        x = self.model.forward(state)
        if self.recurrent:
            x, hidden = self.gru.forward(x.unsqueeze(0), hidden.unsqueeze(0))
            x = x.squeeze(0)
            hidden = hidden.squeeze(0)
        mean = self.action_head.forward(x)
        value = self.value_head.forward(x)
        return (mean, self.logstd), hidden, value.squeeze(-1)
