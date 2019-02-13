"""
Utilities for building vocabulary for onto dataset. We need to return the following
1. A vocabulary dictionary mapping from word to index. Add <unk> and <pad> to the vocabulary as well.
2. A label dictionary mapping label to index and mapping from index to label using list.
3. For pad, we use zero initialization. Don't include them in final loss also. For <unk>, use average embedding
4. maximum_sequence_length

"""

import os
import pickle

from utils.feature.common import sent2tokens, sent2labels, sent2pos
from .data_converter import read_data
from .feature.common import UNKNOWN, PAD


def build_vocab():
    filename = 'pretrain/onto_vocab.pkl'
    if not os.path.isfile(filename):
        print('Building vocab using data/onto.train')
        sentences = read_data('data/onto.train')
        training_sentences = [sent2tokens(sent) for sent in sentences]
        vocab = sorted(list({token for sent in training_sentences for token in sent}))
        vocab.insert(0, UNKNOWN)
        vocab.insert(0, PAD)
        word_index = {word: i for i, word in enumerate(vocab)}

        training_labels = [sent2labels(sent) for sent in sentences]
        all_labels = sorted(list({label for sent in training_labels for label in sent}))
        labels_index = {label: i for i, label in enumerate(all_labels)}

        pos_data = [sent2pos(sent) for sent in sentences]
        all_pos = sorted(list({pos for sent in pos_data for pos in sent}))
        all_pos.insert(0, PAD)
        pos_index = {pos: i for i, pos in enumerate(all_pos)}
        with open(filename, 'wb') as f:
            pickle.dump({
                'vocab': vocab,
                'word_index': word_index,
                'all_labels': all_labels,
                'labels_index': labels_index,
                'all_pos': all_pos,
                'pos_index': pos_index
            }, f)
    else:
        print('Loading vocab from {}'.format(filename))
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        vocab = data['vocab']
        word_index = data['word_index']
        all_labels = data['all_labels']
        labels_index = data['labels_index']
        all_pos = data['all_pos']
        pos_index = data['pos_index']

    return vocab, word_index, all_labels, labels_index, all_pos, pos_index
