from utils.data_converter import read_data
from classifier import CRFClassifier
from utils.feature.manual import ManualFeatureExtractor

def main_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Argument parser for Name Entity Recognition')
    parser.add_argument('method')
    return parser


if __name__ == '__main__':
    train = False
    checkpoint_path = 'checkpoint/crf.model'
    manual_feature_extractor = ManualFeatureExtractor()
    classifier = CRFClassifier(manual_feature_extractor, None)

    if train:
        sentences = read_data('data/onto.train')

        total_num_sentences = len(sentences)
        print('Total number of sentences: {}'.format(total_num_sentences))


        train_sentences = sentences[:int(total_num_sentences * 0.75)]
        val_sentences = sentences[int(total_num_sentences * 0.75):]

        classifier.fit(train_sentences, val_sentences, num_epoch=100, verbose=True)
        classifier.save_checkpoint(checkpoint_path)


    else:
        test_a_sentences = read_data('data/onto.testa')
        classifier.load_checkpoint(checkpoint_path)
        result = classifier.evaluate(test_a_sentences)


