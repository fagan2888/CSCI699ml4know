# Sentiment Analysis and Reinforcement Learning for Portfolio Management
## Steps
1. Finetune bert model on stocknews dataset.
2. Fixed the model and train RL agent to invest on DJIA.
3. Extract real-time twitter and 
4. Need some hand-labeled data.

## Models
1. We need a tokenizer.
2. We need a vocabulary for word embedding.

## Data Source
Dow Johns Index Average from 2008 to 2016.

## Dependencies
[My own Pytorch library torchlib](https://github.com/vermouth1992/torchlib/tree/440257d003c3981a6c25eb4377a7eb295416f61b)
Please clone the project and put in the same folder in project
Other requirements in requirements.txt
pip install -r requirements.txt

## Train Multi-sized CNN
python main_classifier.py
The training log is in multi_size_cnn.log. This requires GPU as it uses Bert Embedding.

## Train Dual-attention 
python main_dual_attention.py --regression --window_size 3
Change the window size to 1, 3, 5, 7.

## Train PPO model
python main_baseline.py

## Test PPO model
python main_baseline.py --test

## Test Trust-region model
In playground.ipynb. It is a little bit messy. But you can run in sequence.