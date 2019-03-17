# -*- coding: utf-8 -*-

import os
import pprint
import time

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import opt
from models import get_model
from models.loss import get_loss_classifier
from preprocess_data import read_relation2id
from semeval import SEMData
from utils.scorer import evaluate


def now():
    return str(time.strftime('%Y-%m-%d %H:%M%S'))


def train(model, index_to_label, loss_fn, classifier_fn):
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # loading data
    train_data = SEMData(opt.data_root, data_type='train')
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    val_data = SEMData(opt.data_root, data_type='val')
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    print('train data: {}; test data: {}'.format(len(train_data), len(val_data)))

    # criterion and optimizer
    # lr = opt.lr
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
        model.cuda()

    #  optimizer = optim.Adam(model.out_linear.parameters(), lr=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)

    # decay learning rate by 0.1 every 50 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_loss = np.inf
    best_f1 = 0.0
    # train
    for epoch in range(opt.num_epochs):

        total_loss = 0.0
        scheduler.step()

        for ii, data in enumerate(tqdm(train_data_loader)):
            if opt.use_gpu:
                data = list(map(lambda x: Variable(x.cuda()), data))
            else:
                data = list(map(Variable, data))

            model.zero_grad()
            out = model.forward(data[:-1])
            loss = loss_fn(out, data[-1])
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()

        train_avg_loss = total_loss / len(train_data_loader.dataset)
        acc, rec, f1, eval_avg_loss, pred_y = eval(model, val_data_loader, index_to_label, loss_fn, classifier_fn)
        if best_f1 < f1:
            best_loss = eval_avg_loss
            best_f1 = f1
            model.save(name="SEM_CNN")
        # toy_acc, toy_f1, toy_loss = eval(model, train_data_loader, opt.rel_num)
        print(
            'Epoch {}/{}: train loss: {:.4f}; test accuracy: {:.4f}, test recall: {:.4f}, test f1:{:.4f},  test loss {:.4f}'.format(
                epoch + 1, opt.num_epochs, train_avg_loss, acc, rec, f1, eval_avg_loss))

    print("*" * 30)
    print("the best f1: {};".format(best_f1))


def eval(model, test_data_loader, index_to_label, loss_fn, classifier_fn):
    model.eval()
    avg_loss = 0.0
    pred_y = []
    labels = []
    for ii, data in enumerate(test_data_loader):

        if opt.use_gpu:
            data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
        else:
            data = list(map(lambda x: torch.LongTensor(x), data))

        out = model.forward(data[:-1])
        label = data[-1]
        loss = loss_fn(out, label)

        predicted = classifier_fn(out)

        pred_y.extend(predicted.cpu().numpy().tolist())
        labels.extend(data[-1].data.cpu().numpy().tolist())
        avg_loss += loss.data.item()

    size = len(test_data_loader.dataset)
    assert len(pred_y) == size and len(labels) == size
    # f1 = f1_score(labels, pred_y, average='micro')
    # acc = accuracy_score(labels, pred_y)

    labels = [index_to_label[idx] for idx in labels]
    pred_y = [index_to_label[idx] for idx in pred_y]
    p, r, f1 = evaluate(labels, pred_y)

    model.train()
    return p, r, f1, avg_loss / size, pred_y


def predict(model, test_data_loader, index_to_label, classifier_fn):
    model.eval()
    pred_y = []
    for ii, data in enumerate(test_data_loader):
        if opt.use_gpu:
            data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
        else:
            data = list(map(lambda x: torch.LongTensor(x), data))

        out = model.forward(data[:-1])
        predicted = classifier_fn(out)
        pred_y.extend(predicted.cpu().numpy().tolist())

    pred_y = [index_to_label[idx] for idx in pred_y]

    model.train()
    return pred_y


def write_result(model_name, pred_y):
    os.makedirs('result', exist_ok=True)
    out = open('result/sem_{}_result.txt'.format(model_name), 'w')
    size = len(pred_y)
    start = 8001
    end = start + size
    for i in range(start, end):
        out.write("{}\t{}\n".format(i, pred_y[i - start]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Argument parser for Relation Extraction')
    parser.add_argument('--model_name', choices=['PCNN', 'PCNNTwoHead', 'PCNNRankLoss'], required=True)

    args = vars(parser.parse_args())
    pprint.pprint(args)

    model_name = args['model_name']

    model = get_model(model_name)(opt)

    if model_name == 'PCNN':
        loss_type = 'cross_entropy'
    elif model_name == 'PCNNTwoHead':
        loss_type = 'two_step'
    elif model_name == 'PCNNRankLoss':
        loss_type = 'rank'
    else:
        raise NotImplementedError

    label_to_index = read_relation2id()
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    os.makedirs('checkpoint', exist_ok=True)

    loss_fn, classifier_fn = get_loss_classifier(loss_type)

    train(model, index_to_label, loss_fn, classifier_fn)

    model.load('checkpoint/{}_SEM_CNN.pth'.format(model_name))
    test_data = SEMData(opt.data_root, data_type='test')
    test_data_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    pred_y = predict(model, test_data_loader, index_to_label, classifier_fn)
    write_result(model.model_name, pred_y)
