#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from chainer import Variable, FunctionSet, optimizers
import chainer.functions  as F
import network
import logging

# 多層パーセプトロンモデル
class MLP(network.Network):

    def __init__(self, num_inputs, num_units, dropout_ratio, optimizer, gpu):
        model = FunctionSet(
            layer1=F.Linear(num_inputs, num_units),
            layer2=F.Linear(num_units, num_units),
            layer3=F.Linear(num_units, 1))  # 回帰用出力
        self.layers = [model.layer1, model.layer2, model.layer3]
        self.dropout_ratio = dropout_ratio
        super(MLP, self).__init__(model, optimizer, gpu)

    # 誤差関数は(ミニバッチ内の)平均二乗誤差
    def forward(self, x_data, y_data, train):
        x = Variable(x_data)
        target = Variable(y_data)
        hidden1 = F.dropout(F.relu(self.model.layer1(x)), ratio=self.dropout_ratio, train=train)
        hidden2 = F.dropout(F.relu(self.model.layer2(hidden1)), ratio=self.dropout_ratio, train=train)
        estimation  = self.model.layer3(hidden2)
        # ２値を返すこと、float32であることを確保することが必要
        #return numpy.array(loss, numpy.float32),
        return F.mean_squared_error(target, estimation), estimation

