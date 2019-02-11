UNKNOWN = '<unk>'
PAD = '<pad>'
MAX_LEN = 156


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2labels_index(sent, label_index):
    return [label_index[label] for token, postag, label in sent]


def sent2pos_index(sent, pos_index):
    return [pos_index[pos] for token, pos, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def sent2token_index(sent, word_index):
    token_lst = []
    for token, _, label in sent:
        if token not in word_index:
            token_lst.append(word_index[UNKNOWN])
        else:
            token_lst.append(word_index[token])

    return token_lst


def sent2pos(sent):
    return [postag for token, postag, label in sent]
