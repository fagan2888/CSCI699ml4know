"""
Finetune pretrained bert model
"""

import numpy as np
import torch
from pytorch_pretrained_bert import BertForTokenClassification
from tqdm import tqdm

from .base import BaseClassifier
from .utils import create_data_loader


class BertClassifier(BaseClassifier):
    def __init__(self, feature_extractor, index_to_label, enable_cuda):
        self.index_to_label = index_to_label
        self.feature_extractor = feature_extractor
        self.model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(index_to_label))
        self.enable_cuda = enable_cuda
        if enable_cuda:
            self.model.cuda()

    def fit(self, train_sentences, val_sentences, num_epoch, verbose):
        pass

    def predict(self, sentences):
        input_ids, _, attention_masks = self.feature_extractor(sentences)
        data_loader = create_data_loader((input_ids, attention_masks), batch_size=16, enable_cuda=self.enable_cuda,
                                         shuffle=False)
        predicted_labels = []
        self.model.eval()
        for X, attention_masks in tqdm(data_loader):
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

        sentence_length = [a.index(0.0) for a in attention_masks]

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

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        if not self.enable_cuda:
            map_location = 'cpu'
        else:
            map_location = None
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
