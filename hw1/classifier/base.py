"""
Define base classifier that all subclass must follow
"""

class BaseClassifier(object):
    def fit(self, train_sentences, test_sentences):
        """ To be consistent, these data needs to be raw data. It's the classifier's job
            to split train/val and perform feature extraction and preprocessing
        """
        raise NotImplementedError

    def evaluate(self, sentences):
        raise NotImplementedError

    def predict(self, sentences):
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_path):
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path):
        raise NotImplementedError