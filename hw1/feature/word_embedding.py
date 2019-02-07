"""
Glove and BERT features
"""

import numpy as np

def load_glove6B(dimension):
    """ Load Glove 6B embedding

    Args:
        dimension: word vector dimension

    Returns:

    """
    glove_file = 'pretrain/glove.6B/glove.6B.{}d.txt'.format(dimension)
    try:
        with open(glove_file, 'r') as f:
            print('Loading {}. Note that this may take a while'.format(glove_file))
            model = {}
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                model[word] = embedding
            print("Done. {} words loaded!".format(len(model)))

    except:
        raise ValueError('Please check whether you have downloaded the Glove embedding. If no, download it at '
              'http://nlp.stanford.edu/data/glove.6B.zip and put it under pretrain/')

    return model
