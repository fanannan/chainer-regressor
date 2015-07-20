#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from chainer import cuda, Variable, FunctionSet
import chainer.functions  as F
import logging

# Chainerによるネットワークの基底クラス
class Network(object):

    # 初期化
    def __init__(self, model, optimizer, dropout_ratio, corruption_level, gpu):
        self.model = model
        #self.layers = []
        self.dropout_ratio = dropout_ratio
        self.corruption_level = corruption_level
        self.optimizer = optimizer
        self.optimizer.setup(self.model.collect_parameters())
        self.rng = np.random.RandomState(1)
        self.gpu = gpu
        if gpu>=0:
            cuda.init(gpu)
            model.to_gpu()

    # ノイズ追加
    def get_corrupted_inputs(self, x_data, train=True):
	    if train:
            # x_data * np.random.binomial(1, 1 - loss_param, len(x_data[0]))
		    ret = self.rng.binomial(size=x_data.shape, n=1, p=1.0-self.corruption_level) * x_data
		    return ret.astype(np.float32)
	    else:
		    return x_data

    # 誤差関数の基本形
    # 回帰問題用なので、原則として、(ミニバッチ内の)平均二乗誤差
    # 継承したクラスでは、２値を返すこと、ひとつ目の値がfloat32であることを確保することが必要
    # return numpy.array(loss, numpy.float32),
    def forward(self, x_data, y_data, train):
        noised_x_data = self.get_corrupted_inputs(x_data, train) 
        estimation  = self.estimate(noised_x_data, train)
        target = Variable(y_data)
        return F.mean_squared_error(target, estimation), estimation

    # 画像化用ウェイト取得
	def get_weight(self, n):
		return self.layers[n].W

    # バッチデータ作成取得
    def get_batch(self, z, index, batch_size, train):
        # self.permはnumpy.arrayなので乱択でバッチデータを生成できる
        # 時系列データの場合は、乱択にするのが良いとは限らないことに注意
        batch = z[self.perm[index:index+batch_size]] if train else z[index:index+batch_size]
        if self.gpu>=0:
            batch = cuda.to_gpu(batch)
        return batch

    # 学習および推定実行
    def learn(self, data, batch_size, train):
        (x, y) = data # 説明変数と非説明変数に分離
        size = len(y)
        assert size == len(x)
        preds = []  # 推定結果保存用
        sum_loss = 0
        if train:   # 学習の場合には、乱数によるインデックスの配列を用意しておく
            self.perm = np.random.permutation(size)
        # 各バッチを処理
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
                # 推定実行
                loss, estimation = self.forward(x_batch, y_batch, train=train)
            # 誤差評価用意
            sum_loss += float(cuda.to_cpu(loss.data)) * batch_size
            # 結果保存
            preds.extend(cuda.to_cpu(estimation.data))
        # 誤差評価
        mean_loss = sum_loss / size
        # 経過出力
        (num_samples, num_features) = np.asarray(preds).shape
        if train or num_features != 1:
            logging.info('mean loss={}'.format(mean_loss))
        else:
            preds_array = np.asarray(preds).reshape(num_samples,)
            y_array = np.asarray(y).reshape(num_samples,)
            pearson = np.corrcoef(preds_array, y_array)
            logging.info('mean loss={}, corrcoef={}'.format(mean_loss, pearson[0][1]))
        return mean_loss

    # モデル構築
    def run(self, train_data, test_data, batch_size, num_epoch, test=False, callback=False):
        mean_losses = []
        for epoch in xrange(1, num_epoch+1):
            logging.info('epoch {}'.format(epoch))
            mean_loss = self.learn(train_data, batch_size, True)    # 学習実行
            if test:
                mean_loss = self.learn(test_data, batch_size, False)    # テスト実行
            if callback:    # 進捗チャート描画
                mean_losses.append(mean_loss)
                callback(mean_losses)
        logging.info('modling completed')
        return None


