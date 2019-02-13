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

    checkpoint_path = 'checkpoint/bert.ckpt'

    classifier.load_checkpoint(checkpoint_path)

    total_num_sentences = len(sentences)
    print('Total number of sentences: {}'.format(total_num_sentences))

    train_sentences = sentences[:int(total_num_sentences * 0.75)]
    val_sentences = sentences[int(total_num_sentences * 0.75):]

    # classifier.fit(train_sentences, val_sentences, num_epoch=5, verbose=True)

    precision, recall, f1_score = classifier.evaluate(val_sentences)

    print('Val prec {} - Val rec {} - Val f1 {}'.format(precision, recall, f1_score))


    testa_sentences = read_data('data/onto.testa')

    precision, recall, f1_score = classifier.evaluate(testa_sentences)

    print('Testa prec {} - Testa rec {} - Testa f1 {}'.format(precision, recall, f1_score))

    classifier.save_checkpoint(checkpoint_path)
