#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from chainer import Variable, FunctionSet, optimizers
import chainer.functions  as F
import network
import logging

# オートエンコーダー 
class DenoisingAutoEncoder(network.Network):

    def __init__(self, num_inputs, num_units, dropout_ratio, corruption_level, optimizer, gpu):
        model = FunctionSet(
            encode=F.Linear(num_inputs, num_units),
            decode=F.Linear(num_units, num_inputs),) 
        self.layers = [model.encode, model.decode]
        self.dropout_ratio = dropout_ratio
        self.corruption_level = corruption_level
        self.rng = np.random.RandomState(1)
        super(DenoisingAutoEncoder, self).__init__(model, optimizer, gpu)

    # ノイズ追加
    def get_corrupted_inputs(self, x_data, train=True):
	    if train:
		    ret = self.rng.binomial(size=x_data.shape, n=1, p=1.0-self.corruption_level) * x_data
		    return ret.astype(np.float32)
	    else:
		    return x_data
		    
    # エンコード後デコードして一致度を追求する
    def forward(self, x_data, _, train):
        noised_x_data = self.get_corrupted_inputs(x_data, train) # x_data * np.random.binomial(1, 1 - loss_param, len(x_data[0]))
        m = self.encode(noised_x_data, train)
        estimation = F.sigmoid(self.model.decode(m))
        target = Variable(x_data)
        return F.mean_squared_error(target, estimation), estimation

    def encode(self, x_data, train):
        x = Variable(x_data)
        m = F.sigmoid(self.model.encode(x))
        #m = F.dropout(F.sigmoid(self.model.encode(x)), ratio=self.dropout_ratio, train=train)
        return m

