"""
Define base classifier that all subclass must follow
"""

import utils.conlleval as conlleval
from utils.data_converter import append_column, data_to_output


class BaseClassifier(object):
    def fit(self, train_sentences, val_sentences, num_epoch, verbose):
        """ To be consistent, these data needs to be raw data. It's the classifier's job
            to split train/val and perform feature extraction and preprocessing
        """
        raise NotImplementedError

    def evaluate(self, sentences):
        y_pred = self.predict(sentences)
        new_sents = append_column(sentences, y_pred)
        precision, recall, f1_score = conlleval.my_evaluate(data_to_output(new_sents))
        return precision, recall, f1_score

    def predict(self, sentences):
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_path):
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path):
        raise NotImplementedError
