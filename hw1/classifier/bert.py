"""
Finetune pretrained bert model
"""

import numpy as np
import torch
import torch.optim as optim
from pytorch_pretrained_bert import BertForTokenClassification
from tqdm import tqdm

import utils.conlleval as conlleval
from utils.data_converter import append_column, data_to_output
from .base import BaseClassifier
from .utils import create_data_loader


class BertClassifier(BaseClassifier):
    def __init__(self, feature_extractor, index_to_label, enable_cuda):
        self.index_to_label = index_to_label
        self.feature_extractor = feature_extractor
        self.model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(index_to_label))
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=3e-5)
        self.enable_cuda = enable_cuda
        if enable_cuda:
            self.model.cuda()

    def fit(self, train_sentences, val_sentences, num_epoch=5, verbose=True):
        train_input_ids, train_label, train_attention_masks = self.feature_extractor(train_sentences)
        train_sentences_length = [len(s) for s in train_sentences]
        val_input_ids, val_label, val_attention_masks = self.feature_extractor(val_sentences)
        val_sentences_length = [len(s) for s in val_sentences]

        train_data_loader = create_data_loader((train_input_ids, train_attention_masks, train_label), batch_size=16,
                                               enable_cuda=self.enable_cuda, shuffle=True)
        train_data_loader_no_shuffle = create_data_loader((train_input_ids, train_attention_masks, train_label),
                                                          batch_size=16, enable_cuda=self.enable_cuda, shuffle=False)
        val_data_loader = create_data_loader((val_input_ids, val_attention_masks, val_label), batch_size=16,
                                             shuffle=False)

        for i in range(num_epoch):
            max_grad_norm = 1.0
            print('Epoch: {}/{}'.format(i + 1, num_epoch))
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for batch in tqdm(train_data_loader):
                self.model.zero_grad()
                b_input_ids, b_input_mask, b_labels = batch
                # forward pass
                loss = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                # backward pass
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                # update parameters
                self.optimizer.step()

            train_acc, train_recall, train_f1 = self._evaluate(train_data_loader_no_shuffle,
                                                               train_sentences_length,
                                                               train_sentences)
            val_acc, val_rec, val_f1 = self._evaluate(val_data_loader, val_sentences_length, val_sentences)

            print(
                'Train acc {:.4f} - Train rec {:.4f} - Train f1 {:.4f} - Val acc {:.4f} - Val rec {:.4f} - Val f1 {:.4f}'.format(
                    train_acc, train_recall, train_f1, val_acc, val_rec, val_f1))

    def _evaluate(self, data_loader, sentence_length, sentences):
        y_pred = self._predict(data_loader, sentence_length)
        new_sents = append_column(sentences, y_pred)
        precision, recall, f1_score = conlleval.my_evaluate(data_to_output(new_sents))
        return precision, recall, f1_score

    def _predict(self, data_loader, sentence_length):
        predicted_labels = []
        self.model.eval()
        for data in tqdm(data_loader):
            X, attention_masks = data[0], data[1]
            X = X.type(torch.FloatTensor)
            if self.enable_cuda:
                X = X.type(torch.cuda.LongTensor)
                attention_masks = attention_masks.type(torch.cuda.FloatTensor)
            else:
                X = X.type(torch.LongTensor)
                attention_masks = attention_masks.type(torch.FloatTensor)

            with torch.no_grad():
                output = self.model.forward(X, token_type_ids=None, attention_mask=attention_masks).cpu().numpy()
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

    def predict(self, sentences):
        input_ids, _, attention_masks = self.feature_extractor(sentences)
        sentence_length = [len(s) for s in sentences]
        data_loader = create_data_loader((input_ids, attention_masks), batch_size=16, enable_cuda=self.enable_cuda,
                                         shuffle=False)
        return self._predict(data_loader, sentence_length)

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        if not self.enable_cuda:
            map_location = 'cpu'
        else:
            map_location = None
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
