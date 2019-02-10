"""
RNN models and loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(self, rnn_type, embedding_matrix, additional_feature_dim, n_tags):
        super(RNNModel, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        if rnn_type == 'bilstm':
            bidirectional = True
            num_direction = 2
        else:
            bidirectional = False
            num_direction = 1
        self.rnn = nn.LSTM(input_size=embedding_dim + additional_feature_dim, hidden_size=256, num_layers=1, bias=True,
                           batch_first=True, dropout=0.2, bidirectional=bidirectional)

        self.linear = nn.Linear(256 * num_direction, n_tags)

    def forward(self, sentences, features=None):
        """

        Args:
            features (Tensor): (batch_size, max_len, feature_dim)
            sentences (Tensor): (batch_size, max_len)

        Returns:
            predicted labels (Tensor): (batch_size * max_len, num_tags)

        """
        sentences_embedding = self.embedding.forward(sentences)
        if features is not None:
            features = torch.cat((sentences_embedding, features), dim=-1)
        else:
            features = sentences_embedding
        output, _ = self.rnn.forward(features)
        output = output.view(-1, output.shape[2])
        output = self.linear.forward(output)

        return F.log_softmax(output, dim=-1)


def loss_fn(outputs, labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # mask out 'PAD' tokens
    mask = (labels >= 0).float()

    # the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).data[0])

    # pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels] * mask

    # cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs) / num_tokens
