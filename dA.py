#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from chainer import Variable, FunctionSet, optimizers
import chainer.functions as F
import network
import logging

# オートエンコーダー 
class DenoisingAutoEncoder(network.Network):

    def __init__(self, num_inputs, num_units, dropout_ratio, corruption_level, optimizer, gpu):
        model = FunctionSet(
            encode=F.Linear(num_inputs, num_units),
            decode=F.Linear(num_units, num_inputs),) 
        self.layers = [model.encode, model.decode]
        super(DenoisingAutoEncoder, self).__init__(model, optimizer, dropout_ratio, corruption_level, gpu)

    # エンコード後デコードして一致度を追求する
    def forward(self, x_data, _, train):
        noised_x_data = self.get_corrupted_inputs(x_data, train) 
        m = self.encode(noised_x_data, train)
        estimation = F.sigmoid(self.model.decode(m))
        target = Variable(x_data)
        return F.mean_squared_error(target, estimation), estimation

    def encode(self, x_data, train):
        x = Variable(x_data)
        m = F.sigmoid(self.model.encode(x))
        #m = F.dropout(F.sigmoid(self.model.encode(x)), ratio=self.dropout_ratio, train=train)
        return m

