# Overview

This code repository provides the data  and evaluation script to help you start your project.

## Data

Both train, testa and (unlabeled) testb are provided. It's stored in `data/` folder. The data is organized into three columns with each representing: token, POS-tag, label

## Evaluation

The offical evaluation script is written in `perl`, which is hard to integrate with current systems. [Here](https://github.com/spyysalo/conlleval.py) is a re-written version of evaluation script in python, which I included in `utils/conlleval.py`. The evaluation script works by taking a test file with *four* columns -- the original file with one additional column of predicted label at the end, it will generate a detailed report available for examination. 

To run this evaluation script, run command line:

`python conlleval.py ${PREDICTION_FILE}`

Or invoke function `evaluate()` directly on data represeted as list of sentences. See `utils/data_converter.py` for more details.
