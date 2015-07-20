#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This code is a Chainer example to train a multi-layer perceptron with diabetes dataset,
based on the code by mottodora (https://gist.github.com/mottodora/a9c46754cf555a68edb7)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from chainer import optimizers
import logging
import reader

def select_optimizer(name):
    if name == "AdaGrad":
        optimizer = optimizers.AdaGrad(lr=0.001)
    elif name == "Adam":
        optimizer = chainer.optimizers.Adam(alpha=0.0001)
    elif name == "MomentumSGD":
        optimizer = optimizers.MomentumSGD(lr=0.01)
    elif name == "RMSprop":
        optimizer = optimizers.RMSprop(lr=0.01)
    elif name == "SGD":
        optimizer = optimizers.SGD(lr=0.01)
    elif name == "AdaDelta":
        optimizer = optimizers.AdaDelta(rho=0.9)
    else:
        raise Exception("Unknown network optimizer: "+args.optimizer)
    return optimizer

def run(name, network, train_data, test_data, batch_size, num_epoch, test=False, chart=True):
    if chart:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        def draw_process_chart(xs):
            ax.clear()
            ax.plot(xs, '-')
            fig.canvas.draw()
            plt.draw()
    #
    network.run(train_data, test_data, batch_size, num_epoch, test=test, callback=draw_process_chart if chart else False)
    if chart:
        plt.savefig('./process_'+name+'.png')
    network.learn(train_data, batch_size, train=True)
    network.learn(test_data, batch_size, train=False)
    if chart:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        actual = map(lambda x:x[0], train_data[1])
        estimated = map(lambda x:x[0], network.estimate(train_data[0], False).data)
        plt.scatter(actual, estimated, marker='o')
        plt.draw()
        plt.savefig('./result_'+name+'.png')
    return network

def execute(name,  train_data, test_data, batch_size, num_inputs, optimizer, num_epoch, gpu, chart):
    if name == "MLP":
        import MLP
        num_units   = 30 # 中間層の数
        dropout_ratio = 0.5 # ドロップアウトの確率
        network = MLP.MLP(num_inputs, num_units, dropout_ratio, optimizer, gpu)
        run(name, network, train_data, test_data, batch_size, num_epoch, test=True, chart=chart)
    elif name == "dA":
        # 次元削減
        import dA
        num_base_units = 7 # 中間層の数
        base_dropout_ratio = 0.5 # ドロップアウトの確率
        base_corruption_level = 0.1 # ゼロノイズの確率
        base_network = dA.DenoisingAutoEncoder(num_inputs, num_base_units, base_dropout_ratio, base_corruption_level, optimizer, gpu)
        base_network = run(name+'_pretune', base_network, train_data, test_data, batch_size, num_epoch, test=False, chart=chart)
        base_x_train = base_network.encode(train_data[0])
        base_x_test = base_network.encode(test_data[0])
        # 回帰
        import MLP
        num_top_units = 12 # 中間層の数
        top_dropout_ratio = 0.05 # ドロップアウトの確率
        top_network = MLP.MLP(base_x_train.shape[1], num_top_units, top_dropout_ratio, optimizer, gpu)
        run(name+'_finetune', top_network, (base_x_train, train_data[1]), (base_x_test, test_data[1]), batch_size, num_epoch, test=True, chart=chart)
    else:
        raise Exception("Unknown network structure: "+args.network)

# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batch_size = 13
# 訓練用データの数をバッチサイズの定数倍で決める
train_size = batch_size * 30
# 学習の繰り返し回数
num_epoch   = 50

if __name__ == '__main__':
    # 引数
    parser = argparse.ArgumentParser(description='Chainer example')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--network', '-n', default="MLP", type=str, help='Network structure')
    parser.add_argument('--optimizer', '-o', default="AdaDelta", type=str, help='Network optimizer')
    parser.add_argument('--normalize', '-nm', default=False, type=bool, help='Apply normalizing to [0, 1]')
    parser.add_argument('--scale', '-s', default=False, type=bool, help='Apply scaling with mean and standard deviation')
    parser.add_argument('--chart', '-c', default=True, type=bool, help='Draw and save charts')
    args = parser.parse_args()
    # 確認進捗表示用
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # データ読み取り
    (train_data, test_data, info) = reader.read_dataset(train_size, normalize=args.normalize, scale=args.scale)
    num_inputs = info["SHAPE_TRAIN_X"][1]
    # モデル実行
    optimizer = select_optimizer(args.optimizer)
    execute(args.network, train_data, test_data, batch_size, num_inputs, optimizer, num_epoch, args.gpu, args.chart)

