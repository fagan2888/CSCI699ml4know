"""
Feature extractors. We consider the following extractors:
1. Hand-crafted features that is similar to https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html.
   This is primarily used for CRF. We need to turn dict features into vector in order to be used in NN.
2. Pre-trained word embedding. We consider
   - Glove
   - State-of-art BERT (Hopefully, this will yield best result)
3. Directly return index of word and index of postag and train word vectors end-to-end. For this part, we will
   choose the best model we get from above. Or initialize the word vector with Glove or BERT.
"""

from .common import sent2token_index, sent2tokens, sent2pos_index, sent2pos, sent2labels_index, sent2labels
from .manual import ManualFeatureExtractor
from .word_embedding import build_embedding_matrix, BertFeatureExtractor, RNNFeatureExtractor
