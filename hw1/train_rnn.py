"""
Train RNN models
"""

from utils.data_converter import read_data
from utils.feature.word_embedding import load_glove6B, build_embedding_matrix, make_word_embedding_feature_extractor
from utils.vocab import build_vocab

if __name__ == '__main__':
    sentences = read_data('data/onto.train')
    vocab, word_index, labels, labels_index = build_vocab(sentences)

    glove_model = load_glove6B()
    embedding_matrix = build_embedding_matrix(glove_model, vocab)
    feature_extractor = make_word_embedding_feature_extractor(word_index, labels_index)
    total_num_sentences = len(sentences)
    print('Total number of sentences: {}'.format(total_num_sentences))

    train_sentences = sentences[:int(total_num_sentences * 0.75)]
    val_sentences = sentences[int(total_num_sentences * 0.75):]
