# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/19 21:10
@Author ：KI 
@File ：text_classification.py
@Motto：Hungry And Humble

"""
import os
import torch
from torch import optim

from data_process import process_sentence, load_cab_vector, process_batch
from models import LSTM
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # load
    print('loading...')
    epoch_num = 10
    arr_train = np.load('npys/arr_train.npy', allow_pickle=True)
    labels_train = np.load('npys/labels_train.npy', allow_pickle=True)
    print('training...')
    model = LSTM(hidden_size=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    criterion = nn.CrossEntropyLoss().to(device)
    loss = 0
    for i in range(epoch_num):
        for j in range(400):
            x = arr_train[j]
            y = labels_train[j]
            # print(y)
            input_ = torch.tensor(x, dtype=torch.float32).to(device)
            label = torch.tensor(y, dtype=torch.long).to(device)
            output = model(input_)
            # print(output)
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print('epoch:%d loss:%.5f' % (i, loss.item()))
    # save model
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, 'models/LSTM.pkl')


def test():
    print('loading...')
    arr_test = np.load('npys/arr_test.npy', allow_pickle=True)
    labels_test = np.load('npys/labels_test.npy', allow_pickle=True)
    print('testing...')
    model = LSTM(hidden_size=64).to(device)
    model.load_state_dict(torch.load('models/LSTM.pkl')['model'])
    model.eval()
    num = 0
    for i in range(100):
        xx = arr_test[i]
        yy = labels_test[i]
        input_ = torch.tensor(xx, dtype=torch.float32).to(device)
        label = torch.tensor(yy, dtype=torch.long).to(device)
        output = model(input_)
        pred = output.max(dim=-1)[1]
        for k in range(100):
            if pred[k] == label[k]:
                num += 1

    print('Accuracy：', num / 10000)


if __name__ == '__main__':
    # train()
    test()
    # load_cab_vector()
    # process_sentence('train')
    # process_sentence('test')
    # process_batch(100)
