"""
Utilities for building vocabulary for onto dataset. We need to return the following
1. A vocabulary dictionary mapping from word to index. Add <unk> and <pad> to the vocabulary as well.
2. A label dictionary mapping label to index and mapping from index to label using list.
3. For pad, we use zero initialization. Don't include them in final loss also. For <unk>, use average embedding
4. maximum_sequence_length

"""

from utils.feature.common import sent2tokens, sent2labels

from .feature.common import UNKNOWN, PAD


def build_vocab(sentences):
    training_data = sentences
    training_sentences = [sent2tokens(sent) for sent in training_data]
    vocab = sorted(list({token for sent in training_sentences for token in sent}))
    vocab.insert(0, UNKNOWN)
    vocab.insert(0, PAD)
    word_index = {word: i for i, word in enumerate(vocab)}

    training_labels = [sent2labels(sent) for sent in training_data]
    labels = sorted(list({label for sent in training_labels for label in sent}))
    labels_index = {label: i for i, label in enumerate(labels)}

    return vocab, word_index, labels, labels_index
