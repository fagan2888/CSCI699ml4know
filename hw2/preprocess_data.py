"""
Read and preprocess data.
"""

import os

import nltk
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_relation2id(filepath='support/relation2id.txt'):
    label_to_index = {}
    with open(filepath, 'r') as f:
        for i in range(19):
            line = f.readline()
            index, label = line.split()
            label_to_index[label] = int(index)
    return label_to_index


def process_sentence_line(line):
    sent_id, sentence = line.split('\t')
    sentence = sentence[1:-2]  # remove quot \n
    e1_left, e1_right = sentence.split('<e1>')
    e1_phrase, e2_sentence = e1_right.split('</e1>')
    between, e2_right = e2_sentence.split('<e2>')
    e2_phrase, e2_right = e2_right.split('</e2>')

    e1_left_sen_tokens = nltk.word_tokenize(e1_left)
    e1_left_index = len(e1_left_sen_tokens)
    e1_tokens = nltk.word_tokenize(e1_phrase)
    e1_right_index = e1_left_index + len(e1_tokens)
    between_tokens = nltk.word_tokenize(between)
    e2_left_index = e1_right_index + len(between_tokens)
    e2_tokens = nltk.word_tokenize(e2_phrase)
    e2_right_index = e2_left_index + len(e2_tokens)
    e2_right_tokens = nltk.word_tokenize(e2_right)

    all_tokens = e1_left_sen_tokens + e1_tokens + between_tokens + e2_tokens + e2_right_tokens

    return sent_id, e1_left_index, e1_right_index - 1, e2_left_index, e2_right_index - 1, all_tokens


def rewrite_semEval2010_task8(infolder='data/', outfolder='support', label_file='support/relation2id.txt'):
    """ Rewrite SemEval2010 dataset. Each line is
    label_index, <e1> index, </e1> index, <e2> index, </e2> index, sentence.

    Also output the sentence to another file to pretrain word embedding. Each line contains a sentence.
    This file contains all the sentences in both training and testing.

    Args:
        infolder: data folder
        train: training data or testing data

    Returns:

    """

    label_to_index = read_relation2id(label_file)

    corpus = []

    print('Processing training data')
    filepath = os.path.join(infolder, 'SemEval2010_task8_training', 'TRAIN_FILE.TXT')
    output = []
    with open(filepath, 'r') as f:
        for _ in tqdm(range(8000)):
            sentence_line = f.readline()
            sent_id, e1_left_index, e1_right_index, e2_left_index, e2_right_index, all_tokens = \
                process_sentence_line(sentence_line)
            # lower all tokens
            all_tokens = [token.lower() for token in all_tokens]

            label = f.readline()[:-1]  # remove \n
            label_index = label_to_index[label]

            label_index = str(label_index)
            e1_left_index = str(e1_left_index)
            e1_right_index = str(e1_right_index)
            e2_left_index = str(e2_left_index)
            e2_right_index = str(e2_right_index)
            all_tokens = " ".join(all_tokens)

            corpus.append(all_tokens)

            f.readline()  # comments
            f.readline()  # empty line

            output.append(" ".join([sent_id, label_index, e1_left_index, e1_right_index, e2_left_index,
                                    e2_right_index, all_tokens]))

    train, val = train_test_split(output, test_size=0.01, random_state=1)
    with open(os.path.join(outfolder, 'train.txt'), 'w') as f:
        f.write("\n".join(train))
        f.write("\n")
    with open(os.path.join(outfolder, 'val.txt'), 'w') as f:
        f.write("\n".join(val))
        f.write("\n")

    print('Processing testing data')
    filepath = os.path.join(infolder, 'SemEval2010_task8_testing', 'TEST_FILE.txt')
    output = []
    with open(filepath, 'r') as f:
        for _ in tqdm(range(2717)):
            sentence_line = f.readline()
            sent_id, e1_left_index, e1_right_index, e2_left_index, e2_right_index, all_tokens = \
                process_sentence_line(sentence_line)
            # lower all tokens
            all_tokens = [token.lower() for token in all_tokens]

            label_index = -1

            label_index = str(label_index)
            e1_left_index = str(e1_left_index)
            e1_right_index = str(e1_right_index)
            e2_left_index = str(e2_left_index)
            e2_right_index = str(e2_right_index)
            all_tokens = " ".join(all_tokens)

            corpus.append(all_tokens)

            output.append(" ".join([sent_id, label_index, e1_left_index, e1_right_index, e2_left_index,
                                    e2_right_index, all_tokens]))

    output_file = os.path.join(outfolder, 'test.txt')
    with open(output_file, 'w') as f:
        f.write("\n".join(output))
        f.write("\n")

    with open(os.path.join(outfolder, 'corpus.txt'), 'w') as f:
        f.write("\n".join(corpus))
        f.write("\n")


if __name__ == '__main__':
    rewrite_semEval2010_task8()
