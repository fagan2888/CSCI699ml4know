"""
Hand-crafted features. Code adapted from https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html.
It returns a list of list of dictionaries.
"""

from .common import sent2labels, FeatureExtractor


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def word2features(sent, i, feature_param):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'word': word.lower(),
        'bias': 1.0,
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
    }

    if 'isupper' in feature_param:
        features['word.isupper()'] = word.isupper()
    if 'istitle' in feature_param:
        features['word.istitle()'] = word.istitle()
    if 'isdigit' in feature_param:
        features['word.isdigit()'] = word.isdigit()
    if 'isfloat' in feature_param:
        features['word.isfloat()'] = is_float(word)
    if 'hyphen' in feature_param:
        features['word.hyphen()'] = '-' in word
    if 'postag' in feature_param:
        features['word.postag'] = postag
        features['postag[:2]'] = postag[:2]

    if 'context' in feature_param:
        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features['-1:word.lower()'] = word1.lower()
            if 'isupper' in feature_param:
                features['-1:word.isupper()'] = word1.isupper()
            if 'istitle' in feature_param:
                features['-1:word.istitle()'] = word1.istitle()
            if 'isdigit' in feature_param:
                features['-1：word.isdigit()'] = word1.isdigit()
            if 'isfloat' in feature_param:
                features['-1:word.isfloat()'] = is_float(word1)
            if 'hyphen' in feature_param:
                features['-1:word.hyphen()'] = '-' in word1
            if 'postag' in feature_param:
                features['-1:word.postag'] = postag1
                features['-1:postag[:2]'] = postag1[:2]
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features['+1:word.lower()'] = word1.lower()

            if 'isupper' in feature_param:
                features['+1:word.isupper()'] = word1.isupper()
            if 'istitle' in feature_param:
                features['+1:word.istitle()'] = word1.istitle()
            if 'isdigit' in feature_param:
                features['+1：word.isdigit()'] = word1.isdigit()
            if 'isfloat' in feature_param:
                features['+1:word.isfloat()'] = is_float(word1)
            if 'hyphen' in feature_param:
                features['+1:word.hyphen()'] = '-' in word1
            if 'postag' in feature_param:
                features['+1:word.postag'] = postag1
                features['+1:postag[:2]'] = postag1[:2]
        else:
            features['EOS'] = True

    return features


def sent2features(sent, feature_param):
    return [word2features(sent, i, feature_param) for i in range(len(sent))]


class ManualFeatureExtractor(FeatureExtractor):
    def __init__(self, feature_param=None):
        if feature_param is None:
            self.feature_param = {}
        else:
            self.feature_param = feature_param

    def __call__(self, sentences):
        X = [sent2features(s, self.feature_param) for s in sentences]
        word_tuple_length = sentences[0][0]
        if len(word_tuple_length) == 3:
            y = [sent2labels(s) for s in sentences]
        elif len(word_tuple_length) == 2:
            y = None
        else:
            raise ValueError(
                'Each word in sent must be (token, postag, label) or (token postag), but got length {}'.format(
                    word_tuple_length))
        return X, y
