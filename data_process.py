# -*- coding:utf-8 -*-
"""
@Time：2022/06/10 10:45
@Author：KI
@File：data_process.py
@Motto：Hungry And Humble
"""
import os
import re

import numpy as np


def load_data(path, flag='train'):
    labels = ['pos', 'neg']
    data = []
    for label in labels:
        files = os.listdir(os.path.join(path, flag, label))
        #
        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
        for file in files:
            with open(os.path.join(path, flag, label, file), 'r', encoding='utf8') as rf:
                temp = rf.read().replace('\n', '')
                temp = temp.replace('<br /><br />', ' ')
                temp = re.sub(r, '', temp)
                temp = temp.split(' ')
                temp = [temp[i].lower() for i in range(len(temp)) if temp[i] != '']
                if label == 'pos':
                    data.append([temp, 1])
                elif label == 'neg':
                    data.append([temp, 0])
    return data


def process_sentence(flag):
    sentence_code = []
    vocabulary_vectors = np.load('npys/vocabulary_vectors.npy', allow_pickle=True)
    word_list = np.load('npys/word_list.npy', allow_pickle=True)
    word_list = word_list.tolist()
    test_data = load_data('Imdb', flag)
    for i in range(len(test_data)):
        # print(i)
        vec = test_data[i][0]
        temp = []
        index = 0
        for j in range(len(vec)):
            try:
                index = word_list.index(vec[j])
            except ValueError:  # not find
                index = 399999
            finally:
                temp.append(index)  # index
        if len(temp) < 250:
            for k in range(len(temp), 250):  # append 0
                temp.append(0)
        else:
            temp = temp[0:250]  # 只保留250个
        sentence_code.append(temp)

    # print(sentence_code)

    sentence_code = np.array(sentence_code)
    if flag == 'train':
        np.save('npys/sentence_code_1', sentence_code)
    else:
        np.save('npys/sentence_code_2', sentence_code)


# define word list
def load_cab_vector():
    word_list = []
    vocabulary_vectors = []
    data = open('glove.6B.50d.txt', encoding='utf-8')
    for line in data.readlines():
        temp = line.strip('\n').split(' ')
        name = temp[0]
        word_list.append(name.lower())
        vector = [temp[i] for i in range(1, len(temp))]
        vector = list(map(float, vector))
        vocabulary_vectors.append(vector)
    # save
    vocabulary_vectors = np.array(vocabulary_vectors)
    word_list = np.array(word_list)
    np.save('npys/vocabulary_vectors', vocabulary_vectors)
    np.save('npys/word_list', word_list)
    return vocabulary_vectors, word_list


def process_batch(batch_size):
    test_data = load_data('Imdb', flag='test')
    train_data = load_data('Imdb')
    sentence_code_1 = np.load('npys/sentence_code_1.npy', allow_pickle=True)
    sentence_code_1 = sentence_code_1.tolist()
    # 25000 * 250
    sentence_code_2 = np.load('npys/sentence_code_2.npy', allow_pickle=True)
    sentence_code_2 = sentence_code_2.tolist()
    vocabulary_vectors = np.load('npys/vocabulary_vectors.npy', allow_pickle=True)
    vocabulary_vectors = vocabulary_vectors.tolist()

    for i in range(25000):
        sentence_code_1[i] = [vocabulary_vectors[x] for x in sentence_code_1[i]]
        sentence_code_2[i] = [vocabulary_vectors[x] for x in sentence_code_2[i]]
        # for j in range(250):
        #     sentence_code_1[i][j] = vocabulary_vectors[sentence_code_1[i][j]]
        #     sentence_code_2[i][j] = vocabulary_vectors[sentence_code_2[i][j]]
    data = train_data + test_data
    sentence_code = np.r_[sentence_code_1, sentence_code_2]
    # shuffle
    shuffle_ix = np.random.permutation(np.arange(len(data)))
    data = np.array(data)[shuffle_ix].tolist()
    sentence_code = sentence_code[shuffle_ix]

    train_data = data[:int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):]
    sentence_code_1 = sentence_code[:int(len(sentence_code) * 0.8)]
    sentence_code_2 = sentence_code[int(len(sentence_code) * 0.8):]

    labels_train = []
    labels_test = []
    arr_train = []
    arr_test = []

    # mini-batch
    for i in range(1, int(len(train_data) / batch_size) + 1):
        arr_train.append(sentence_code_1[(i - 1) * batch_size:i * batch_size])
        labels_train.append([train_data[j][1] for j in range((i - 1) * batch_size, i * batch_size)])
    for i in range(1, int(len(test_data) / batch_size) + 1):
        arr_test.append(sentence_code_2[(i - 1) * batch_size:i * batch_size])
        labels_test.append([test_data[j][1] for j in range((i - 1) * batch_size, i * batch_size)])

    arr_train = np.array(arr_train)
    arr_test = np.array(arr_test)
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)
    np.save('npys/arr_train', arr_train)
    np.save('npys/arr_test', arr_test)
    np.save('npys/labels_train', labels_train)
    np.save('npys/labels_test', labels_test)

    return arr_train, labels_train, arr_test, labels_test
