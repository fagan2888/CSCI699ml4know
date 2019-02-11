from utils.vocab import build_vocab
from utils.data_converter import read_data
from utils.feature.word_embedding import BertFeatureExtractor
from classifier.bert import BertClassifier

if __name__ == '__main__':
    enable_cuda = True
    sentences = read_data('data/onto.train')
    vocab, word_index, index_to_label, labels_index, all_pos, pos_index = build_vocab(sentences)

    feature_extractor = BertFeatureExtractor(labels_index)

    classifier = BertClassifier(feature_extractor, index_to_label, enable_cuda)

    classifier.load_checkpoint('checkpoint/bert.ckpt')

    total_num_sentences = len(sentences)
    print('Total number of sentences: {}'.format(total_num_sentences))

    train_sentences = sentences[:int(total_num_sentences * 0.75)]
    val_sentences = sentences[int(total_num_sentences * 0.75):]

    precision, recall, f1_score = classifier.evaluate(val_sentences)



