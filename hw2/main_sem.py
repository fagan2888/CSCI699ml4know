# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import opt
from models import PCNNTwoHead
from preprocess_data import read_relation2id
from semeval import SEMData
from utils.scorer import evaluate


def now():
    return str(time.strftime('%Y-%m-%d %H:%M%S'))


def two_step_loss(score, label):
    """ Separate the "other" label. Use a binary classifier to detect "other" label and another
    18-way classifier to detect true labels.

    Args:
        true_label_score: tensor of shape (batch_size, 18)
        other_label_score: tensor of shape (batch_size, 2)
        label: (batch_size)

    Returns: loss

    """
    true_label_score, other_label_score = score
    not_other = label != 18
    binary_loss = F.cross_entropy(other_label_score, not_other.type(torch.LongTensor))

    # for class loss, we only consider those which is not "Other"
    true_label_score_not_other = true_label_score[not_other]
    not_other_label = label[not_other]
    class_loss = F.cross_entropy(true_label_score_not_other, not_other_label)
    return binary_loss + class_loss


def get_loss(type='cross_entropy'):
    if type == 'cross_entropy':
        return F.cross_entropy

    elif type == 'two_step':
        return two_step_loss

    else:
        raise NotImplementedError


def cross_entropy_classifier(score):
    return torch.max(score, 1)[1].data


def two_step_classifier(score):
    true_label_score, other_label_score = score
    label = torch.max(true_label_score, 1)[1].data
    other_label = torch.max(other_label_score, 1)[1].data
    label[other_label == 0] = 18
    return label


def get_classifier(type='cross_entropy'):
    if type == 'cross_entropy':
        return cross_entropy_classifier
    elif type == 'two_step':
        return two_step_classifier


def train(index_to_label, loss_fn, classifier_fn):
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
    model = PCNNTwoHead(opt)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
        model.cuda()

    criterion = get_loss(loss_type)
    #  optimizer = optim.Adam(model.out_linear.parameters(), lr=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)

    best_loss = np.inf
    best_f1 = 0.0
    # train
    for epoch in range(opt.num_epochs):

        total_loss = 0.0

        for ii, data in enumerate(tqdm(train_data_loader)):
            if opt.use_gpu:
                data = list(map(lambda x: Variable(x.cuda()), data))
            else:
                data = list(map(Variable, data))

            model.zero_grad()
            out = model.forward(data[:-1])
            loss = criterion(out, data[-1])
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()

        train_avg_loss = total_loss / len(train_data_loader.dataset)
        acc, rec, f1, eval_avg_loss, pred_y = eval(model, val_data_loader, index_to_label, loss_fn, classifier_fn)
        if eval_avg_loss < best_loss:
            best_loss = eval_avg_loss
            best_f1 = f1
            # write_result(model.model_name, pred_y)
            model.save(name="SEM_CNN")
        # toy_acc, toy_f1, toy_loss = eval(model, train_data_loader, opt.rel_num)
        print(
            'Epoch {}/{}: train loss: {:.4f}; test accuracy: {:.4f}, test recall: {:.4f}, test f1:{:.4f},  test loss {:.4f}'.format(
                epoch + 1, opt.num_epochs, train_avg_loss, acc, rec, f1, eval_avg_loss))

    print("*" * 30)
    print("the best f1: {};".format(best_f1))
    return model


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
    label_to_index = read_relation2id()
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    os.makedirs('checkpoint', exist_ok=True)

    loss_type = 'two_step'
    loss_fn = get_loss(loss_type)
    classifier_fn = get_classifier(loss_type)

    model = train(index_to_label, loss_fn, classifier_fn)

    model.load('checkpoint/PCNN_SEM_CNN.pth')
    test_data = SEMData(opt.data_root, data_type='test')
    test_data_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    pred_y = predict(model, test_data_loader, index_to_label, classifier_fn)
    write_result(model.model_name, pred_y)
