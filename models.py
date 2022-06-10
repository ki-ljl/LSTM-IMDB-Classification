# -*- coding:utf-8 -*-
"""
@Time：2022/06/10 10:43
@Author：KI
@File：models.py
@Motto：Hungry And Humble
"""
from torch import nn


class LSTM(nn.Module):
    def __init__(self, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=50, hidden_size=hidden_size, num_layers=1,
                            batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                # nn.Linear(hidden_size, 32),
                                nn.Linear(hidden_size, 2),
                                nn.ReLU())

    def forward(self, input_seq):
        # print(x.size())
        x, _ = self.lstm(input_seq)
        x = x[:, -1, :]
        x = self.fc(x)
        #
        return x
