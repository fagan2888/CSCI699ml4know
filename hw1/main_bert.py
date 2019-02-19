import pprint

from sklearn.model_selection import train_test_split

from classifier.bert import BertClassifier
from utils import enable_cuda
from utils.data_converter import read_data
from utils.feature.word_embedding import BertFeatureExtractor
from utils.vocab import build_vocab


def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for training Bert model')

    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-ne', '--num_epoch', type=int, default=20)
    train_parser.add_argument('-c', '--continue', action='store_true')
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser('eval', help='If the test file contains label, print performance report, '
                                                     'else output a file with labels')
    eval_parser.add_argument('-f', '--infile')
    eval_parser.add_argument('-o', '--outfile')
    eval_parser.set_defaults(func=eval)

    return parser


def build_classifier():
    _, _, index_to_label, labels_index, _, _ = build_vocab()
    feature_extractor = BertFeatureExtractor(labels_index)
    classifier = BertClassifier(feature_extractor, index_to_label, enable_cuda)
    return classifier


def get_checkpoint_path():
    checkpoint_path = 'checkpoint/bert.ckpt'
    return checkpoint_path


def train(args):
    args = vars(args)
    pprint.pprint(args)
    num_epoch = args['num_epoch']
    sentences = read_data('data/onto.train')

    total_num_sentences = len(sentences)
    print('Total number of sentences: {}'.format(total_num_sentences))
    train_sentences, val_sentences = train_test_split(sentences, test_size=0.25, random_state=123, shuffle=True)

    classifier = build_classifier()
    if args['continue']:
        classifier.load_checkpoint(get_checkpoint_path())
    classifier.fit(train_sentences, val_sentences, num_epoch=num_epoch, verbose=True)
    classifier.save_checkpoint(get_checkpoint_path())


def eval(args):
    args = vars(args)
    pprint.pprint(args)
    infile = args['infile']
    classifier = build_classifier()
    classifier.load_checkpoint(get_checkpoint_path())
    test_sentences = read_data(infile)
    precision, recall, f1_score = classifier.evaluate(test_sentences)
    print('Test prec {} - Test rec {} - Test f1 {}'.format(precision, recall, f1_score))


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)
