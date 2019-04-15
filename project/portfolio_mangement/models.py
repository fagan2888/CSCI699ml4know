"""
Implement classification models.
We treat the stock price prediction problem as a time series classification problem.
Given a window of price and driving series (news), predict the next day, 15 days later and 1 month later.
First, we use a simple open source sentiment analyzer tool to extract the polarity score of each sequence.
(The news is actually very noisy, so a simple model may be good enough for this purpose)
Then, we apply Dual-stage attention model for time series classification. (https://arxiv.org/pdf/1704.02971.pdf)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    def __init__(self, hidden_size, dimension):
        super(AttentionModule, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_size * 2, dimension), requires_grad=True)
        self.U = nn.Parameter(torch.randn(dimension, dimension), requires_grad=True)
        self.v = nn.Parameter(torch.randn(dimension, 1), requires_grad=True)
        self.hidden_size = hidden_size

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, hidden_state, cell_state, input):
        """

        Args:
            hidden_state: (batch_size, hidden_size)
            cell_state: (batch_size, hidden_size)
            input: Tensor of shape (batch_size, num_samples, dimension)

        Returns: score of (batch_size, num_samples)

        """
        hidden = torch.cat((hidden_state, cell_state), dim=-1)  # (batch_size, 2 * hidden_size)
        input_weight = torch.matmul(input, self.U)  # (batch_size, num_samples, dimension)
        hidden_weight = torch.matmul(hidden, self.W).unsqueeze(1)  # (batch_size, 1, dimension)
        logits = torch.matmul(torch.tanh(input_weight + hidden_weight), self.v).squeeze(-1)  # (batch_size, num_samples)
        score = F.softmax(logits, dim=-1)  # (batch_size, num_samples)
        return score


class DualAttentionRNN(nn.Module):
    def __init__(self, input_hidden_size, temporal_hidden_size, num_driving_series, window_size, regression=False):
        super(DualAttentionRNN, self).__init__()
        self.input_attention = AttentionModule(hidden_size=input_hidden_size, dimension=window_size)
        self.temporal_attention = AttentionModule(hidden_size=temporal_hidden_size, dimension=input_hidden_size)
        self.encoder_lstm = nn.LSTMCell(num_driving_series, input_hidden_size)
        self.decoder_lstm = nn.LSTMCell(1, temporal_hidden_size)

        self.decoder_y_hat = nn.Linear(input_hidden_size + 1, 1)

        if regression:
            self.output_linear = nn.Linear(temporal_hidden_size, 1)
        else:
            self.output_linear = nn.Linear(temporal_hidden_size, 2)

        self.num_driving_series = num_driving_series
        self.window_size = window_size
        self.input_hidden_size = input_hidden_size
        self.temporal_hidden_size = temporal_hidden_size

    def forward(self, history, driving_series):
        """ T is the window size. At time step t, we observe history price from day t - T + 1, ... t.
            The goal is to predict whether price(t + 1) > price(t) and output the probability.

        Args:
            history: (batch_size, T)
            driving_series: (batch_size, N, T)

        Returns: probability of whether tomorrow's price will be higher. (batch_size, 2)

        """
        batch_size = history.shape[0]
        input_hx = torch.zeros(batch_size, self.input_hidden_size)
        input_cx = torch.zeros(batch_size, self.input_hidden_size)
        input_hidden_out = []
        for i in range(self.window_size):
            weights = self.input_attention.forward(input_hx, input_cx, driving_series)
            input = weights * driving_series[:, :, i]
            input_hx, input_cx = self.encoder_lstm.forward(input, (input_hx, input_cx))
            input_hidden_out.append(input_hx)

        input_hidden_out = torch.stack(input_hidden_out, dim=1)  # (batch_size, T, input_hidden_size)

        output_hx = torch.zeros(batch_size, self.temporal_hidden_size)
        output_cx = torch.zeros(batch_size, self.temporal_hidden_size)
        for i in range(self.window_size):
            weights = self.temporal_attention.forward(output_hx, output_cx, input_hidden_out)  # (batch_size, T)
            c_i = torch.sum(weights.unsqueeze(-1) * input_hidden_out, dim=1)  # (batch_size, input_hidden_size)
            y_i = torch.cat((c_i, history[:, i:i + 1]), dim=-1)  # (batch_size, input_hidden_size + 1)
            y_i_hat = self.decoder_y_hat.forward(y_i)  # (batch_size, 1)
            output_hx, output_cx = self.decoder_lstm.forward(y_i_hat, (output_hx, output_cx))

        prediction = self.output_linear.forward(output_hx)

        return prediction
