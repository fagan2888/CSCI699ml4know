UNKNOWN = '<unk>'
PAD = '<pad>'
MAX_LEN = 75


def sent2labels(sent):
    return [s[-1] for s in sent]


def sent2labels_index(sent, label_index):
    return [label_index[s[-1]] for s in sent]


def sent2pos_index(sent, pos_index):
    return [pos_index[s[1]] for s in sent]


def sent2tokens(sent):
    return [s[0] for s in sent]


def sent2token_index(sent, word_index):
    token_lst = []
    for s in sent:
        token = s[0]
        if token not in word_index:
            token_lst.append(word_index[UNKNOWN])
        else:
            token_lst.append(word_index[token])

    return token_lst


def sent2pos(sent):
    return [postag for token, postag, label in sent]


class FeatureExtractor(object):
    def __call__(self, sentences):
        raise NotImplementedError
