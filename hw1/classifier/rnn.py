"""
RNN based classifier. We will try some variances including:
1. GRU, LSTM and BiLSTM
2. Add CNN layer on top of LSTM (Try directly CNN-based classification)
3. Train embedding or fix embedding
4. Try various embedding including Glove and BERT
5. Add additional features: word-level features and character-level features
6. Try different objective. (Cross entropy vs Macro-F1)
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import utils.conlleval as conlleval
from utils.data_converter import append_column, data_to_output
from .base import BaseClassifier
from .utils import create_data_loader


class RNNClassifier(BaseClassifier):
    def __init__(self, feature_extractor, index_to_label, rnn_model: nn.Module, optimizer, loss_fn, enable_cuda=False,
                 scheduler=None):
        """

        Args:
            feature_extractor (func): function to extractor features (postag). These features will be concatenated with
                               embedding
            index_to_label (list): index to label. Used in prediction
            rnn_cell_type (str): 'LSTM', 'GRU', 'BiLSTM'
            embedding_type (str): 'Glove', 'Skipgram', etc. Only support context-free
            freeze_embedding (bool): whether the embedding is trainable.
            loss_fn (nn.Module): loss function
        """
        self.index_to_label = index_to_label
        self.model = rnn_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.loss_fn = loss_fn
        self.enable_cuda = enable_cuda
        if enable_cuda:
            self.model.cuda()

    def fit(self, train_sentences, val_sentences, num_epoch, verbose=True):
        X_train, X_train_feature, y_train, train_sentences_length = self.feature_extractor(train_sentences)
        X_val, X_val_feature, y_val, val_sentences_length = self.feature_extractor(val_sentences)
        train_data_loader = create_data_loader((X_train, X_train_feature, y_train), batch_size=128,
                                               enable_cuda=self.enable_cuda, shuffle=True)
        train_data_loader_no_shuffle = create_data_loader((X_train, X_train_feature, y_train), batch_size=128,
                                                          enable_cuda=self.enable_cuda, shuffle=False)
        val_data_loader = create_data_loader((X_val, X_val_feature, y_val), batch_size=128,
                                             enable_cuda=self.enable_cuda, shuffle=False)

        for i in range(num_epoch):
            print('Epoch: {}/{}'.format(i + 1, num_epoch))
            total_loss = 0.0
            total = 0
            for X, feature, y in tqdm(train_data_loader):
                if self.enable_cuda:
                    X = X.type(torch.cuda.LongTensor)
                    feature = feature.type(torch.cuda.FloatTensor)
                    y = y.type(torch.cuda.LongTensor)

                else:
                    X = X.type(torch.LongTensor)
                    feature = feature.type(torch.FloatTensor)
                    y = y.type(torch.LongTensor)

                self.optimizer.zero_grad()
                output = self.model.forward(X, feature)
                loss = self.loss_fn(output, y)
                total_loss += loss.item() * y.size(0)
                total += y.size(0)
                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            train_acc, train_recall, train_f1 = self._evaluate(train_data_loader_no_shuffle,
                                                               train_sentences_length,
                                                               train_sentences)
            val_acc, val_rec, val_f1 = self._evaluate(val_data_loader, val_sentences_length, val_sentences)

            print(
                'Train acc {:.4f} - Train rec {:.4f} - Train f1 {:.4f} - Val acc {:.4f} - Val rec {:.4f} - Val f1 {:.4f}'.format(
                    train_acc, train_recall, train_f1, val_acc, val_rec, val_f1))

    def _predict(self, data_loader, sentence_length):
        predicted_labels = []
        self.model.eval()
        for data in tqdm(data_loader):
            X, feature = data[0], data[1]
            X = X.type(torch.FloatTensor)
            if self.enable_cuda:
                X = X.type(torch.cuda.LongTensor)
                feature = feature.type(torch.cuda.FloatTensor)
            else:
                X = X.type(torch.LongTensor)
                feature = feature.type(torch.FloatTensor)

            with torch.no_grad():
                output = self.model.forward(X, feature).cpu().numpy()
                output = np.argmax(output, axis=-1)
                predicted_labels.append(output)
        self.model.train()

        predicted_labels = np.concatenate(predicted_labels, axis=0)

        assert len(sentence_length) == predicted_labels.shape[0], 'Length must match'
        final_labels = []
        for i in range(predicted_labels.shape[0]):
            current_result = []
            if sentence_length[i] <= predicted_labels.shape[1]:
                for j in range(sentence_length[i]):
                    current_result.append(self.index_to_label[predicted_labels[i][j]])
            else:
                for j in range(predicted_labels.shape[1]):
                    current_result.append(self.index_to_label[predicted_labels[i][j]])
                for _ in range(sentence_length[i] - predicted_labels.shape[1]):
                    current_result.append('O')
            final_labels.append(current_result)

        return final_labels

    def _evaluate(self, data_loader, sentence_length, sentences):
        y_pred = self._predict(data_loader, sentence_length)
        new_sents = append_column(sentences, y_pred)
        precision, recall, f1_score = conlleval.my_evaluate(data_to_output(new_sents))
        return precision, recall, f1_score

    def predict(self, sentences):
        X, feature, _, sentence_length = self.feature_extractor(sentences)
        data_loader = create_data_loader((X, feature), batch_size=128, enable_cuda=self.enable_cuda, shuffle=False)
        return self._predict(data_loader, sentence_length)

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        if not self.enable_cuda:
            map_location = 'cpu'
        else:
            map_location = None
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
