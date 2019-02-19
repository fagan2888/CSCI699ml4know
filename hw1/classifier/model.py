"""
RNN models and loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """ Self attention Layer """

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        proj_query = self.query_conv(x).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = self.gamma * out + x
        return out


class RNNModel(nn.Module):
    def __init__(self, rnn_type, embedding_matrix, additional_feature_dim, n_tags, n_layers):
        super(RNNModel, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        embedding_matrix = torch.from_numpy(embedding_matrix).type(torch.FloatTensor)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        if rnn_type == 'bilstm':
            bidirectional = True
            num_direction = 2
            self.rnn = nn.LSTM(input_size=embedding_dim + additional_feature_dim, hidden_size=64, num_layers=n_layers,
                               bias=True, dropout=0.5,
                               batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'lstm':
            bidirectional = False
            num_direction = 1
            self.rnn = nn.LSTM(input_size=embedding_dim + additional_feature_dim, hidden_size=64, num_layers=n_layers,
                               bias=True, dropout=0.5,
                               batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'cnn':
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            num_channel = 64
            modules = []
            modules.append(nn.Conv1d(in_channels=embedding_dim + additional_feature_dim, out_channels=num_channel,
                                     kernel_size=kernel_size, padding=padding, stride=1))
            modules.append(SelfAttention(num_channel))
            modules.append(nn.ReLU())
            for _ in range(n_layers - 1):
                modules.append(nn.Conv1d(in_channels=num_channel, out_channels=num_channel,
                                         kernel_size=kernel_size, padding=padding, stride=1))
                modules.append(SelfAttention(num_channel))
                modules.append(nn.ReLU())
            self.rnn = nn.Sequential(*modules)
            num_direction = 1
        else:
            raise ValueError('Unknown rnn_type {}'.format(rnn_type))

        self.linear = nn.Linear(64 * num_direction, n_tags)

        self.rnn_type = rnn_type

    def forward(self, sentences, features=None):
        """

        Args:
            features (Tensor): (batch_size, max_len, feature_dim)
            sentences (Tensor): (batch_size, max_len)

        Returns:
            predicted labels (Tensor): (batch_size * max_len, num_tags)

        """
        sentences_embedding = self.embedding.forward(sentences)
        features = torch.cat((sentences_embedding, features), dim=-1)
        if self.rnn_type == 'cnn':
            features = torch.transpose(features, 1, 2)
            output = self.rnn.forward(features)
            output = torch.transpose(output, 1, 2)
        else:
            output, _ = self.rnn.forward(features)
        batch_size, max_len, feature_dim = output.shape
        output = output.contiguous().view(-1, output.shape[2])
        output = self.linear.forward(output)

        output = output.view(batch_size, max_len, -1)

        return F.log_softmax(output, dim=-1)


def cross_entropy_with_mask(outputs, labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)
    outputs = outputs.view(-1, outputs.shape[2])

    # mask out 'PAD' tokens
    mask = (labels >= 0).float()

    # the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).item())

    # pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels] * mask

    # cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs) / num_tokens
