import pprint

from classifier import CRFClassifier
from utils.data_converter import read_data, append_column, extract_columns, data_to_output
from utils.feature.manual import ManualFeatureExtractor

feature_set = {
    'isupper',
    'istitle',
    'isdigit',
    'isfloat',
    'hyphen',
    'postag',
    'context'
}


def get_checkpoint(feature_list):
    feature_str = '_'.join(sorted(feature_list))
    checkpoint_path = 'checkpoint/crf_{}.model'.format(feature_str)
    return checkpoint_path


def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Name Entity Recognition using Conditional Random Field')

    parser.add_argument('feature', nargs='*', type=str, help='feature set {}'.format(feature_set))

    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-ne', '--num_epoch', type=int, default=100)
    train_parser.add_argument('-c1', '--c1', type=float, default=0.1)
    train_parser.add_argument('-c2', '--c2', type=float, default=0.1)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser('eval', help='If the test file contains label, print performance report, '
                                                        'else output a file with labels')
    eval_parser.add_argument('-f', '--infile')
    eval_parser.set_defaults(func=eval)

    predict_parser = subparsers.add_parser('predict', help='If the test file contains label, print performance report, '
                                                     'else output a file with labels')
    predict_parser.add_argument('-f', '--infile')
    predict_parser.add_argument('-o', '--outfile')
    predict_parser.set_defaults(func=predict)

    return parser


def train(args):
    args = vars(args)
    pprint.pprint(args)

    feature_list = args['feature']
    num_epoch = args['num_epoch']
    c1 = args['c1']
    c2 = args['c2']

    for feature in feature_list:
        assert feature in feature_set, 'Feature {} not in the feature set'.format(feature)

    checkpoint_path = get_checkpoint(feature_list)
    manual_feature_extractor = ManualFeatureExtractor(feature_list)
    classifier = CRFClassifier(manual_feature_extractor, c1=c1, c2=c2)

    sentences = read_data('data/onto.train')

    total_num_sentences = len(sentences)
    print('Total number of sentences: {}'.format(total_num_sentences))

    train_sentences = sentences[:int(total_num_sentences * 0.75)]
    val_sentences = sentences[int(total_num_sentences * 0.75):]

    classifier.fit(train_sentences, val_sentences, num_epoch=num_epoch, verbose=True)

    precision, recall, f1_score = classifier.evaluate(val_sentences)
    print('Result on validation sentence: Precition {}, Recall {}, F1 {}'.format(precision, recall, f1_score))

    test_a_sentences = read_data('data/onto.testa')
    precision, recall, f1_score = classifier.evaluate(test_a_sentences)
    print('Result on onto.testa: Precition {}, Recall {}, F1 {}'.format(precision, recall, f1_score))

    classifier.save_checkpoint(checkpoint_path)


def eval(args):
    args = vars(args)
    pprint.pprint(args)
    infile = args['infile']
    feature_list = args['feature']

    for feature in feature_list:
        assert feature in feature_set, 'Feature {} not in the feature set'.format(feature)

    checkpoint_path = get_checkpoint(feature_list)
    manual_feature_extractor = ManualFeatureExtractor(feature_list)
    classifier = CRFClassifier(manual_feature_extractor)
    classifier.load_checkpoint(checkpoint_path)

    test_sentence = read_data(infile)
    precision, recall, f1_score = classifier.evaluate(test_sentence)
    print('Result on onto.testa: Precition {}, Recall {}, F1 {}'.format(precision, recall, f1_score))



def predict(args):
    args = vars(args)
    pprint.pprint(args)
    infile = args['infile']
    outfile = args['outfile']
    feature_list = args['feature']
    if not feature_list:
        feature_list = []

    for feature in feature_list:
        assert feature in feature_set, 'Feature {} not in the feature set'.format(feature)

    checkpoint_path = get_checkpoint(feature_list)
    manual_feature_extractor = ManualFeatureExtractor(feature_list)
    classifier = CRFClassifier(manual_feature_extractor, None)
    classifier.load_checkpoint(checkpoint_path)

    test_sentence = read_data(infile)
    result = classifier.predict(test_sentence)

    test_sentence_label = extract_columns(append_column(test_sentence, result), indexs=[-1])
    data_to_output(test_sentence_label, write_to_file=outfile)


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)
