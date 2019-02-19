from __future__ import print_function, division

from utils import conlleval


def read_data(fname, ignore_docstart=False):
    """Read data from any files with fixed format.
    Each line of file should be a space-separated token information,
    in which information starts from the token itself.
    Each sentence is separated by a empty line.

    e.g. 'Apple NP (NP I-ORG' could be one line

    Args:
        fname (str): file path for reading data.

    Returns:
        sentences (list):
          Sentences is a list of sentences.
          Sentence is a list of token information.
          Token information is in format: [token, feature_1, ..., feature_n, tag_label]
    """
    sentences, prev_sentence = [], []
    with open(fname) as f:
        for line in f:
            if not line.strip():
                if prev_sentence and (not ignore_docstart or len(prev_sentence) > 1):
                    sentences.append(prev_sentence)
                prev_sentence = []
                continue
            prev_sentence.append(list(line.strip().split()))
        if len(prev_sentence) > 0:
            sentences.append(prev_sentence)
    return sentences


def data_to_output(sentences, write_to_file=''):
    """Convert data to a string of data stream that is ready to write

    Args:
        sentences (list): A list of sentences
        write_to_file (str, optional):
          If a file path, write data stream to that file

    Returns:
        output_list: A list of strings, each line indicating a line in file
    """
    output_list = []
    for sentence in sentences:
        for tup in sentence:
            output_list.append('\t'.join(tup))
        output_list.append('')
    if write_to_file:
        with open(write_to_file, 'w') as f:
            f.write('\n'.join(output_list))
    return output_list


def get_column(sentences, index):
    """Get a column of information from sentences

    Args:
        sentences (list):
          Sentences is a list of sentences.
          Sentence is a list of token information.
          Token information is in format: [token, feature_1, ..., feature_n, tag_label]

        index (int): An index to retrieve from. Can be positive
          or negative (backward)

    Returns:
        columns (list): Same format as sentences.
    """
    columns = []
    for sentence in sentences:
        columns.append([tup[index] for tup in sentence])
    return columns


def extract_columns(sentences, indexs):
    """Extract columns of information from sentences

    Args:
        sentences (list):
          Sentences is a list of sentences.
          Sentence is a list of token information.
          Token information is in format: [token, feature_1, ..., feature_n, tag_label]

        indexs (list): A list of indexs to retrieve from. Can be positive
          or negative (backward)

    Returns:
        columns (list): Same format as sentences.
    """
    columns = []
    for sentence in sentences:
        columns.append([[tup[i] for i in indexs] for tup in sentence])
    return columns


def append_column(sentences, column):
    """Append a column to list of sentences

    Args:
        sentences (list):
          Sentences is a list of sentences.
          Sentence is a list of token information.
          Token information is in format: [token, feature_1, ..., feature_n, tag_label]
        column (list): Same format as sentence

    Returns:
        new_sentences: same format as sentences
    """
    new_sentences = []
    for sentence, col in zip(sentences, column):
        new_sentences.append(
            [tup + [c] for tup, c in zip(sentence, col)]
        )
    return new_sentences


def extend_columns(sentences, columns):
    """Extend column to list of sentences

    Args:
        sentences (list):
          Sentences is a list of sentences.
          Sentence is a list of token information.
          Token information is in format: [token, feature_1, ..., feature_n, tag_label]
        columns (list): Same format as sentence

    Returns:
        new_sentences: same format as sentences
    """
    new_sentences = []
    for sentence, column in zip(sentences, columns):
        new_sentences.append(
            [tup + col for tup, col in zip(sentence, column)]
        )
    return new_sentences


def main():
    sents = read_data('data/onto.testa')
    print(sents[1])
    tags = extract_columns(sents, [-1])
    print(tags[1])
    # copy ground-truth label as the predicted label
    new_sents = extend_columns(sents, tags)
    print(new_sents[1])
    data_to_output(new_sents, write_to_file="out.testa")
    conlleval.evaluate(data_to_output(new_sents))


if __name__ == '__main__':
    main()
