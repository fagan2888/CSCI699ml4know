"""
Train RNN models
"""

import torch.optim as optim

from classifier.model import RNNModel, loss_fn
from classifier.rnn import RNNClassifier
from utils.data_converter import read_data
from utils.feature.word_embedding import load_glove6B, build_embedding_matrix, make_word_embedding_feature_extractor
from utils.vocab import build_vocab

if __name__ == '__main__':
    enable_cuda = False
    sentences = read_data('data/onto.train')
    vocab, word_index, index_to_label, labels_index, all_pos, pos_index = build_vocab(sentences)

    glove_model = load_glove6B()
    embedding_matrix = build_embedding_matrix(glove_model, vocab)
    del glove_model
    feature_extractor = make_word_embedding_feature_extractor(word_index, labels_index, pos_index)
    total_num_sentences = len(sentences)
    print('Total number of sentences: {}'.format(total_num_sentences))

    train_sentences = sentences[:int(total_num_sentences * 0.75)]
    val_sentences = sentences[int(total_num_sentences * 0.75):]

    rnn_model = RNNModel('lstm', embedding_matrix, len(all_pos), len(index_to_label))
    if enable_cuda:
        rnn_model.cuda()

    optimizer = optim.Adam(rnn_model.parameters(), lr=1e-4)

    classifier = RNNClassifier(feature_extractor, index_to_label, rnn_model, optimizer, loss_fn,
                               enable_cuda=enable_cuda)

    classifier.fit(train_sentences, val_sentences, num_epoch=100, verbose=True)
