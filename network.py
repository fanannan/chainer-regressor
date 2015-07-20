#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import logging

class Network(object):

    def __init__(self, model, optimizer, gpu):
        self.model = model
        #self.model = FunctionSet(
        #    layer1=F.Linear(10, n_units),
        #    layer2=F.Linear(n_units, n_units),
        #    layer3=F.Linear(n_units, 1))
        #self.layers = [self.model.layer1, self.model.layer2, self.model.layer3]
        self.optimizer = optimizer
        self.optimizer.setup(self.model.collect_parameters())
        self.gpu = gpu
        if gpu>=0:
            cuda.init(gpu)
            model.to_gpu()

    # 誤差関数は(ミニバッチ内の)平均二乗誤差
    def forward(self, x_data, y_data, train):
        # ２値を返すこと、float32であることを確保することが必要
        #return numpy.array(loss, numpy.float32),
        return numpy.array(0.0, numpy.float32), 

	def get_weight(self, n):
		return self.layers[n].W
		
    def get_batch(self, z, index, batch_size, train):
        # self.permはnumpy.arrayなので乱択でバッチデータを生成できる
        batch = z[self.perm[index:index+batch_size]] if train else z[index:index+batch_size]
        if self.gpu>=0:
            batch = cuda.to_gpu(batch)
        return batch

    def learn(self, data, batch_size, train=True):
        (x, y) = data
        size = len(y)
        preds = []
        sum_loss = 0
        if train:
            self.perm = np.random.permutation(size)
        for index in xrange(0, size, batch_size):
            x_batch = self.get_batch(x, index, batch_size, train)
            y_batch = self.get_batch(y, index, batch_size, train)
            if train:
                # 勾配を初期化
                self.optimizer.zero_grads()
                # 順伝播させて誤差等を算出
                loss, estimation = self.forward(x_batch, y_batch, train=train)
                # 誤差逆伝播で勾配を計算
                loss.backward()
                #self.optimizer.clip_grads(1.0)
                self.optimizer.update()
            else:
                loss, estimation = self.forward(x_batch, y_batch, train=train)
            sum_loss += float(cuda.to_cpu(loss.data)) * batch_size
            preds.extend(cuda.to_cpu(estimation.data))
        mean_loss = sum_loss / size
        if train:
            logging.info('mean loss={}'.format(mean_loss))
        else:
            (num_samples, num_features) = np.asarray(preds).shape
            if num_features == 1:
                preds_array = np.asarray(preds).reshape(num_samples,)
                y_array = np.asarray(y).reshape(num_samples,)
                pearson = np.corrcoef(preds_array, y_array)
                logging.info('mean loss={}, corrcoef={}'.format(mean_loss, pearson[0][1]))
            else:
                logging.info('mean loss={}'.format(mean_loss))
        return mean_loss

    def run(self, train_data, test_data, batch_size, num_epoch, test=False, callback=False):
        # Learning loop
        mean_losses = []
        for epoch in xrange(1, num_epoch+1):
            logging.info('epoch {}'.format(epoch))
            mean_loss = self.learn(train_data, batch_size, True)
            if test:
                mean_loss = self.learn(test_data, batch_size, False)
            if callback:
                mean_losses.append(mean_loss)
                callback(mean_losses)
        logging.info('modling completed')

