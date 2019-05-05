import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from torchlib.common import FloatTensor, LongTensor
from torchlib.deep_rl.utils.distributions import FixedNormal
from torchlib.utils.layers import freeze


class PriceOnlyPolicyModule(nn.Module):
    def __init__(self, nn_size, state_dim, action_dim, recurrent=False, hidden_size=20):
        super(PriceOnlyPolicyModule, self).__init__()
        random_number = np.random.randn(action_dim)

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
        return FixedNormal(mean, self.logstd), hidden, value.squeeze(-1)


class NewsPredictorModule(nn.Module):
    """
    A predictor for news data. The input is indices of 25
    """

    def __init__(self, seq_length, freeze_embedding=False, window_length_lst=(3, 4, 5, 6, 7, 8)):
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
        self.model = nn.ModuleList()
        out_channels = 2
        for window_length in window_length_lst:
            self.model.append(nn.Conv1d(in_channels=768, out_channels=out_channels, kernel_size=window_length,
                                        stride=1, padding=(window_length // 2)))
        self.max_pool = nn.MaxPool1d(kernel_size=64, stride=64)
        self.relu = nn.ReLU()
        linear_size = out_channels * len(window_length_lst) * seq_length // 64

        self.linear = nn.Sequential(
            nn.Linear(linear_size, 2),
        )

    def forward(self, state):
        """

        Args:
            state: shape is (batch_size, num_stocks, seq_length) with idx of words.

        Returns: the probability of stock goes up (batch_size, num_stocks)

        """
        state = state.type(LongTensor)
        batch_size, num_stocks, seq_length = state.shape
        embedding = self.embedding.forward(state)
        embedding = embedding.view(batch_size * num_stocks, seq_length, 768).permute(0, 2, 1)

        cnn_out = []
        for model in self.model:
            out = model.forward(embedding)
            out = self.relu.forward(out)
            out = self.max_pool.forward(out)
            out = out.view(batch_size, -1)
            cnn_out.append(out)  # (batch_size * num_stocks, seq_length // 64, 2) = (batch_size, 25, 2)

        # average with all the news
        cnn_out = torch.cat(cnn_out, dim=-1)  # (batch_size * num_stocks, 50 * len(window))
        prob = self.linear.forward(cnn_out)
        prob = prob.view(batch_size, num_stocks, 2)
        prob = prob.permute(0, 2, 1)
        return prob


class NewsOnlyPolicyModule(nn.Module):
    def __init__(self, num_stocks, seq_length, recurrent=False, hidden_size=16, freeze_embedding=False):
        super(NewsOnlyPolicyModule, self).__init__()
        self.model = NewsPredictorModule(seq_length, freeze_embedding=freeze_embedding)

        action_dim = num_stocks + 1
        random_number = np.random.randn(action_dim)
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
        return FixedNormal(mean, self.logstd), hidden, value.squeeze(-1)


from torchlib.deep_rl import BaseAgent


class TrustRegionAgent(BaseAgent):
    def __init__(self, predictor, threshold=0.7):
        super(TrustRegionAgent, self).__init__()
        self.predictor = predictor
        self.threshold = threshold

    def predict(self, state):
        assert state.shape == (1, 1600)
        state = np.expand_dims(state, axis=0)
        with torch.no_grad():
            out = self.predictor.forward(state).cpu().numpy()
        out = np.squeeze(out, axis=0)  # (nun_stocks, 2)
        probability_up = out[:, 0]
        probability_up[probability_up < self.threshold] = 0
        probability_up = np.insert(probability_up, 0, 0.5)  # insert cash probability
        action = probability_up / np.sum(probability_up)
        return action
