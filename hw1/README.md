# Overview

This code repository provides the data  and evaluation script to help you start your project.

## Data

Both train, testa and (unlabeled) testb are provided. It's stored in `data/` folder. The data is organized into three columns with each representing: token, POS-tag, label

Note that there is a bug in the original read_data, which will ignore the last sentence. In this repo, this function is fixed.

## Evaluation

The offical evaluation script is written in `perl`, which is hard to integrate with current systems. [Here](https://github.com/spyysalo/conlleval.py) is a re-written version of evaluation script in python, which I included in `utils/conlleval.py`. The evaluation script works by taking a test file with *four* columns -- the original file with one additional column of predicted label at the end, it will generate a detailed report available for examination. 

To run this evaluation script, run command line:

`python conlleval.py ${PREDICTION_FILE}`

Or invoke function `evaluate()` directly on data represeted as list of sentences. See `utils/data_converter.py` for more details.


## Package Requirement
See requirement.txt. To install, run command line

`pip install -r requirements.txt`

We use [GloVe](https://nlp.stanford.edu/projects/glove/) to initialize our embedding. To download, run

```bash
cd pretrain
bash download_glove.sh
cd ..
```

Make sure to make directory named checkpoint under hw1 to save models.

`mkdir checkpoint`

The data is removed for copyright issues. You need to put data under hw1. The directory should be

```
hw1/data/onto.testa
hw1/data/onto.testb
hw1/data/onto.train
```

## Conditional Random Field
### Model training

`python main_crf.py isupper istitle isdigit isfloat hyphen postag context train -ne 100`

### Evaluate on onto.testa

`python main_crf.py isupper istitle isdigit isfloat hyphen postag context eval --infile data/onto.testa`

### Predict results for onto.testb

`python main_crf.py isupper istitle isdigit isfloat hyphen postag context predict --infile data/onto.testb --outfile crf_output.txt`

## RNN-based Model
### Model training

`python main_rnn.py -a bilstm -nl 2 -mf -l cross_entropy -d 50 train -lr 1e-3 -ne 20`

### Evaluate on onto.testa

`python main_rnn.py -a bilstm -nl 3 -mf -l cross_entropy -d 50 eval --infile data/onto.testa`

### Predict results for onto.testb

`python main_rnn.py -a bilstm -nl 2 -mf -l cross_entropy -d 50 predict --infile data/onto.testb --outfile rnn_output.txt`

## Bert Finetuning (time-consuming)

`python main_bert.py train -ne 20`

### Evaluate on onto.testa

`python main_bert.py eval --infile data/onto.testa`