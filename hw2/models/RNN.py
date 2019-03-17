"""
Implement RNN based model with Attention in
Zhang (2015). Relation classification via recurrent neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BasicModule import BasicModule


class RNNAttention(BasicModule):
    def __init__(self, opt):
        super(RNNAttention, self).__init__()

        self.model_name = 'RNNAttention'

        self.opt = opt
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.pos1_embs = nn.Embedding(self.opt.pos_size + 1, self.opt.pos_dim)
        self.pos2_embs = nn.Embedding(self.opt.pos_size + 1, self.opt.pos_dim)

        feature_dim = self.opt.word_dim + self.opt.pos_dim * 2

        self.bilstm = nn.LSTM(input_size=feature_dim, hidden_size=32, num_layers=1, batch_first=True,
                              bidirectional=True)

        self.attention_matrix = nn.Parameter(torch.randn(64, 1), requires_grad=True)
        self.dropout = nn.Dropout(self.opt.drop_out)
        self.out_linear = nn.Linear(64, self.opt.rel_num)

    def forward(self, x):
        # we don't use lexical feature in this model

        lexical_feature, word_feautre, left_pf, right_pf = x

        batch_size = lexical_feature.size(0)
        # lexical_level_emb = self.word_embs(lexical_feature)  # (batch_size, 6, word_dim)
        # lexical_level_emb = lexical_level_emb.view(batch_size, -1)
        # lexical_level_emb = lexical_level_emb.sum(1)

        # sentence level feature
        word_emb = self.word_embs(word_feautre)  # (batch_size, max_len, word_dim)
        left_emb = self.pos1_embs(left_pf)  # (batch_size, max_len, pos_dim)
        right_emb = self.pos2_embs(right_pf)  # (batch_size, max_len, pos_dim)

        sentence_feature = torch.cat([word_emb, left_emb, right_emb], 2)  # (batch_size, max_len, word_dim + pos_dim *2)

        output, _ = self.bilstm.forward(sentence_feature)

        M = torch.tanh(output)

        attention_matrix = self.attention_matrix.repeat(batch_size, 1, 1)

        alpha = F.softmax(torch.bmm(M, attention_matrix).squeeze(dim=-1), dim=-1)  # (batch_size, T)

        alpha = alpha.unsqueeze(dim=1)  # (batch_size, 1, T)

        r = torch.bmm(alpha, M).squeeze(dim=1)  # (batch_size, 1, hidden_size) -> (batch_size, hidden_size)

        h = torch.tanh(r)

        h = self.dropout.forward(h)

        score = self.out_linear.forward(h)

        return score


class RNNAttentionRankLoss(RNNAttention):
    def __init__(self, opt):
        opt.rel_num -= 1
        super(RNNAttentionRankLoss, self).__init__(opt)
        self.model_name = 'RNNAttentionRankLoss'
