"""
Wrapper for conditional random field classifier
"""

import joblib
import sklearn_crfsuite

from .base import BaseClassifier


class CRFClassifier(BaseClassifier):
    def __init__(self, feature_extractor, params, verbose=False):
        """

        Args:
            feature_extractor: a function that takes a sentence and return a list of dictionary.
            params: hyperparameter of conditional random field
        """
        self.feature_extractor = feature_extractor
        self.params = params

        self.model = sklearn_crfsuite.CRF(algorithm='lbfgs',
                                          c1=0.1,
                                          c2=0.1,
                                          max_iterations=100,
                                          all_possible_transitions=True,
                                          verbose=verbose)

    def fit(self, train_sentences, val_sentences):
        x_train, y_train = self.feature_extractor(train_sentences)
        x_val, y_val = self.feature_extractor(val_sentences)
        self.model.fit(x_train, y_train, x_val, y_val)

    def predict(self, sentences):
        x, _ = self.feature_extractor(sentences)
        return self.model.predict(x)

    def save_checkpoint(self, checkpoint_path):
        joblib.dump(self.model, checkpoint_path)
        print('Saving model to {}'.format(checkpoint_path))

    def load_checkpoint(self, checkpoint_path):
        self.model = joblib.load(checkpoint_path)
        print('Loading model from {}'.format(checkpoint_path))
