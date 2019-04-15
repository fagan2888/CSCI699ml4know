"""
Sentiment Analyzer for extract sentence sentiment features
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


class BaseSentimentAnalyzer(object):
    def __str__(self):
        raise NotImplementedError

    def predict(self, sentence):
        """ Predict the sentiment score of a sentence. Ranging from [-1, 1],
            where -1 stands for extreme negative and 1 stands for extreme positive.

        Args:
            sentence: a string

        Returns:

        """
        raise NotImplementedError


class NLTKSentimentAnalyzer(BaseSentimentAnalyzer):
    def __init__(self):
        self.sia = SIA()

    def __str__(self):
        return 'nltk'

    def predict(self, sentence):
        try:
            pol_score = self.sia.polarity_scores(sentence)
            return pol_score['compound']
        except:
            return 0.0
