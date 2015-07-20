#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import preprocessing
import logging

# データ読み込み
def read_dataset(train_size, scale=False, normalize=False):
    logging.info('fetching the diabetes dataset')
    diabetes = load_diabetes()
    data = diabetes['data'].astype(np.float32)
    target = diabetes['target'].astype(np.float32).reshape(len(diabetes['target']), 1)
    #"Chainerのmnist.pyだと下記ののような書き方になっているが、ミニバッチの数が2以上だと動かない"らしい 
    #target = diabetes['target'].astype(np.float32) 
    # 本来訓練データで標準化・正規化して、そのパラメータをテストデータに適用すべき
    if normalize and scale:
        raise Exception('both normalize and scale can not be True')
    if normalize:
        data = preprocessing.normalize(data)
        target = preprocessing.normalize(target)
    if scale:
        data = preprocessing.scale(data)
        target = preprocessing.scale(target)
    # 分割
    x_train, x_test = np.split(data, [train_size])
    y_train, y_test = np.split(target, [train_size])
    assert len(x_train)==len(y_train)
    assert len(x_test)==len(y_test)
    return  ((x_train, y_train), (x_test, y_test), 
        {"SHAPE_TRAIN_X":x_train.shape,
          "SHAPE_TRAIN_Y":y_train.shape,
          "SHAPE_TEST_X":x_test.shape,
          "SHAPE_TEST_Y":y_test.shape,
          })

