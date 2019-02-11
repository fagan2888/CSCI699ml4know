"""
Glove and BERT features
"""

import os
import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

from .common import UNKNOWN, PAD, MAX_LEN
from .common import sent2token_index, sent2labels_index, sent2pos_index, sent2tokens
from .common import FeatureExtractor

glove_total = 400000


def load_glove6B(dimension=50):
    """ Load Glove 6B embedding

    Args:
        dimension: word vector dimension

    Returns:

    """
    glove_file = 'pretrain/glove.6B/glove.6B.{}d.txt'.format(dimension)
    pickle_file = 'pretrain/glove.6B/glove.6B.{}d.pkl'.format(dimension)
    if os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as f:
            model = pickle.load(f)
            print('Successfully load {}'.format(pickle_file))
            return model

    average_embedding = 0
    try:
        with open(glove_file, 'r') as f:
            print('Loading {}. Note that this may take a while'.format(glove_file))
            model = {}
            for line in tqdm(f, total=glove_total):
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                average_embedding += embedding
                model[word] = embedding
            print("Done. {} words loaded!".format(len(model)))

    except:
        raise ValueError('Please check whether you have downloaded the Glove embedding. If no, download it at '
                         'http://nlp.stanford.edu/data/glove.6B.zip and put it under pretrain/')

    # use average word vector as unknown word
    average_embedding = average_embedding / glove_total
    model[UNKNOWN] = average_embedding
    model[PAD] = np.zeros_like(average_embedding)

    with open(pickle_file, 'wb') as f:
        pickle.dump(model, f)

    return model


def build_embedding_matrix(embedding_model, vocab, verbose=False):
    """ Building embedding_matrix given model, vocab and word_index. Only applicable to context-free embedding

    Args:
        embedding_model (dict): map from word to embedding.
        vocab (list): a list of vocabulary including unknown and padding
        verbose (bool): verbose mode

    Returns: embedding_matrix (len(vocab), embedding_dim)

    """
    vocab_size = len(vocab)
    embedding_dim = embedding_model[UNKNOWN].shape[0]
    print('Embedding dim: {}'.format(embedding_dim))
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    print('Building embedding matrix')
    missed_word = 0
    for i, word in enumerate(vocab):
        if word == UNKNOWN:
            embedding_matrix[i] = embedding_model[UNKNOWN]
        elif word == PAD:
            embedding_matrix[i] = embedding_model[PAD]
        elif word.lower() in embedding_model:
            embedding_matrix[i] = embedding_model[word.lower()]
        else:
            if verbose:
                print("{} not in pre-trained embedding, use random instead".format(word))
            embedding_matrix[i] = np.random.randn(embedding_dim)
            missed_word += 1
    print("Total word: {}. Missed word: {}. Missing ratio: {}".format(vocab_size,
                                                                      missed_word, missed_word / vocab_size))

    return embedding_matrix




class RNNFeatureExtractor(FeatureExtractor):
    def __init__(self, word_index, labels_index, pos_index):
        self.word_index = word_index
        self.labels_index = labels_index
        self.pos_index = pos_index

    def __call__(self, sentences):
        """ It translate words into index and pad into the same length using PAD

        Args:
            sentences: a list of sentence (a list of tuple with (word, pos, tag))

        Returns: (word_index, additional_feature, tag, mask)

        """
        X = [sent2token_index(s, self.word_index) for s in sentences]
        word_tuple = sentences[0][0]
        if len(word_tuple) == 3:
            y = [sent2labels_index(s, self.labels_index) for s in sentences]
        elif len(word_tuple) == 2:
            y = None
        else:
            raise ValueError(
                'Each word in sent must be (token, postag, label) or (token postag), but got length {}'.format(
                    len(word_tuple)))

        pos = [sent2pos_index(s, self.pos_index) for s in sentences]
        # pad sequence
        X = pad_sequences(X, maxlen=MAX_LEN, padding='post', truncating='post', value=self.word_index[PAD])
        pos_feature = pad_sequences(pos, maxlen=MAX_LEN, padding='post', truncating='post', value=self.pos_index['XX'])
        pos_feature = to_categorical(pos_feature, len(self.pos_index))
        y = pad_sequences(y, maxlen=MAX_LEN, padding='post', truncating='post', value=-1)

        sentence_length = [len(s) for s in sentences]
        return X, pos_feature, y, sentence_length


class BertFeatureExtractor(FeatureExtractor):
    def __init__(self, labels_index, tokenizer=None):
        self.labels_index = labels_index
        if not tokenizer:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            self.tokenizer = tokenizer

    def __call__(self, sentences):
        """ Extractor features to finetune pretrained Bert Model

        Args:
            sentences: list of list of (token, pos, tag)

        Returns:
            token_idx: index of token in bert tokenizer
            tag_index: index of tag
            mask: 1 for non-padding token and 0 for padding token

        """
        joint_sentences = [" ".join(sent2tokens(sent)) for sent in sentences]
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in joint_sentences]

        word_tuple = sentences[0][0]
        if len(word_tuple) == 3:
            y = [sent2labels_index(s, self.labels_index) for s in sentences]
            y = pad_sequences([[self.labels_index.get(l) for l in lab] for lab in y],
                              maxlen=MAX_LEN, value=self.labels_index["O"], padding="post",
                              dtype="long", truncating="post")
        elif len(word_tuple) == 2:
            y = None
        else:
            raise ValueError(
                'Each word in sent must be (token, postag, label) or (token postag), but got length {}'.format(
                    len(word_tuple)))

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


        attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

        return input_ids, y, attention_masks

