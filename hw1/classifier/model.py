"""
RNN models and loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            modules = []
            modules.append(nn.Conv1d(in_channels=embedding_dim + additional_feature_dim, out_channels=64,
                                     kernel_size=5, padding=2, stride=1))
            modules.append(nn.ReLU())
            for _ in range(n_layers - 1):
                modules.append(nn.Conv1d(in_channels=64, out_channels=64,
                                         kernel_size=5, padding=2, stride=1))
                modules.append(nn.BatchNorm1d(num_features=64))
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
