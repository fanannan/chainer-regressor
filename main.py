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

def run(name, network, train_data, test_data, batch_size, num_epoch, pretune, test=False, chart=True):
    draw_process_chart = False
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
    network.run(train_data, test_data, batch_size, num_epoch, test=test, callback=draw_process_chart)
    if chart:
        plt.savefig('./process_'+name+'.png')
    network.learn(train_data, batch_size, train=True)
    network.learn(test_data, batch_size, train=False)
    if chart and not pretune:
        def draw_result_chart(name, data):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.clear()
            estimated = map(lambda x:x[0], network.estimate(data[0], False).data)
            actual = map(lambda x:x[0], data[1])
            plt.scatter(estimated, actual, marker='o')
            plt.draw()
            plt.savefig('./result_'+name+'.png')
        #
        draw_result_chart(name+'_train', train_data)
        draw_result_chart(name+'_test', test_data)
    return network

def execute(name,  train_data, test_data, batch_size, num_inputs, optimizer, num_epoch, gpu, chart):
    if name == "MLP":
        import MLP
        num_units   = 30 # 中間層の数
        dropout_ratio = 0.5 # ドロップアウトの確率
        network = MLP.MLP(num_inputs, num_units, dropout_ratio, optimizer, gpu)
        run(name, network, train_data, test_data, batch_size, num_epoch, False, test=True, chart=chart)
    elif name == "dA":
        # -scを指定して、スケーリングしないとパフォーマンスが出ない模様
        # 次元削減
        import dA
        num_base_units = 7 # 中間層の数
        base_dropout_ratio = 0.5 # ドロップアウトの確率
        base_corruption_level = 0.1 # ゼロノイズの確率
        base_network = dA.DenoisingAutoEncoder(num_inputs, num_base_units, base_dropout_ratio, base_corruption_level, optimizer, gpu)
        base_network = run(name+'_pretune', base_network, train_data, test_data, batch_size, num_epoch, True, test=False, chart=chart)
        base_x_train = base_network.encode(train_data[0], False).data
        base_x_test = base_network.encode(test_data[0], False).data
        # 回帰
        import MLP
        num_top_units = 12 # 中間層の数
        top_dropout_ratio = 0.05 # ドロップアウトの確率
        top_network = MLP.MLP(base_x_train.shape[1], num_top_units, top_dropout_ratio, optimizer, gpu)
        run(name+'_finetune', top_network, (base_x_train, train_data[1]), (base_x_test, test_data[1]), batch_size, num_epoch, False, test=True, chart=chart)
    elif name == "SdA":
        # 次元削減
        import dA
        num_base_units = 7 # 中間層の数
        base_dropout_ratio = 0.5 # ドロップアウトの確率
        base_corruption_level = 0.1 # ゼロノイズの確率
        base_network1 = dA.DenoisingAutoEncoder(num_inputs, num_base_units, base_dropout_ratio, base_corruption_level, optimizer, gpu)
        base_network1 = run(name+'_pretune1', base_network1, train_data, test_data, batch_size, num_epoch, True, test=False, chart=chart)
        base_x_train1 = base_network1.encode(train_data[0], False).data
        base_x_test1 = base_network1.encode(test_data[0], False).data
        base_network2 = dA.DenoisingAutoEncoder(base_x_train1.shape[1], num_base_units, base_dropout_ratio, base_corruption_level, optimizer, gpu)
        base_network2 = run(name+'_pretune2', base_network2, (base_x_train1, train_data[1]), (base_x_test1, test_data[1]), batch_size, num_epoch, True, test=False, chart=chart)
        base_x_train2 = base_network2.encode(base_x_train1, False).data
        base_x_test2 = base_network2.encode(base_x_test1, False).data
        # 回帰
        import MLP
        num_top_units = 12 # 中間層の数
        top_dropout_ratio = 0.05 # ドロップアウトの確率
        top_network = MLP.MLP(base_x_train2.shape[1], num_top_units, top_dropout_ratio, optimizer, gpu)
        run(name+'_finetune', top_network, (base_x_train2, train_data[1]), (base_x_test2, test_data[1]), batch_size, num_epoch, False, test=True, chart=chart)
    else:
        raise Exception("Unknown network structure: "+args.network)


if __name__ == '__main__':
    # 引数
    # 例: python main.py -n dA -sc -c
    parser = argparse.ArgumentParser(description='Chainer example')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--network', '-n', default="MLP", type=str, help='Network structure')
    parser.add_argument('--optimizer', '-o', default="AdaDelta", type=str, help='Network optimizer')
    parser.add_argument('--normalize', '-nm', default=False, action='store_true', help='Apply normalizing to [0, 1]')
    parser.add_argument('--scale', '-sc', default=False, action='store_true', help='Apply scaling with mean and standard deviation')
    parser.add_argument('--chart', '-c', default=False, action='store_true', help='Draw and save charts')
    parser.add_argument('--epoch', '-e', default=100, help='Number of learning epoches')
    parser.add_argument('--batchsize', '-b', default=13, help='Number of records in a batch')
    parser.add_argument('--trainsize', '-t', default=390, help='Number of training records in whole dataset')
    args = parser.parse_args()
    # 確認進捗表示用
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # 訓練用データの数をバッチサイズの定数倍で決める
    train_size = args.trainsize
    # データ読み取り
    (train_data, test_data, info) = reader.read_dataset(train_size, normalize=args.normalize, scale=args.scale)
    num_inputs = info["SHAPE_TRAIN_X"][1] # 入力層の要素数
    # 確率的勾配降下法で学習させる際の１回分のバッチサイズ
    batch_size = args.batchsize
    # 学習の繰り返し回数
    num_epoch = args.epoch
    # オプティマイザー指定
    optimizer = select_optimizer(args.optimizer)
    # モデル実行
    execute(args.network, train_data, test_data, batch_size, num_inputs, optimizer, num_epoch, args.gpu, args.chart)

