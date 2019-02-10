"""
RNN based classifier. We will try some variances including:
1. GRU, LSTM and BiLSTM
2. Add CNN layer on top of LSTM (Try directly CNN-based classification)
3. Train embedding or fix embedding
4. Try various embedding including Glove and BERT
5. Add additional features: word-level features and character-level features
6. Try different objective. (Cross entropy vs Macro-F1)
"""

import torch.nn as nn

from .base import BaseClassifier
from .utils import create_data_loader


class RNNClassifier(BaseClassifier):
    def __init__(self, feature_extractor, rnn_model: nn.Module, optimizer, loss_fn, enable_cuda=False):
        """

        Args:
            feature_extractor (func): function to extractor features (postag). These features will be concatenated with
                               embedding
            rnn_cell_type (str): 'LSTM', 'GRU', 'BiLSTM'
            embedding_type (str): 'Glove', 'Skipgram', etc. Only support context-free
            freeze_embedding (bool): whether the embedding is trainable.
            loss_fn (nn.Module): loss function
        """
        self.model = rnn_model
        self.optimizer = optimizer
        self.feature_extractor = feature_extractor
        self.loss_fn = loss_fn
        self.enable_cuda = enable_cuda
        if enable_cuda:
            self.model.cuda()

    def fit(self, train_sentences, val_sentences):
        X_train, X_train_feature, y_train = self.feature_extractor(train_sentences)
        X_val, X_val_feature, y_val = self.feature_extractor(val_sentences)
        train_data_loader = create_data_loader((X_train, X_train_feature, y_train), batch_size=32,
                                               enable_cuda=self.enable_cuda)
        val_data_loader = create_data_loader((X_val, X_val_feature, y_val), batch_size=32,
                                             enable_cuda=self.enable_cuda)

        self.optimizer.zero_grad()
        self.model.forward()

    def predict(self, sentences):
        pass

    def save_checkpoint(self, checkpoint_path):
        pass

    def load_checkpoint(self, checkpoint_path):
        pass

