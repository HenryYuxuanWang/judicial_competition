# -*- coding: utf-8 -*-
import numpy as np

def label2tag(labels, sets):
    return [sets[i == 1] for i in labels]


def predict2half(predictions, sets):
    m = []
    for x in predictions:
        x_return = sets[x > 0.5]
        m.append(x_return)
    return m


def predict2toptag(predictions, sets):
    m = []
    for x in predictions:
        x_return = sets[x == x.max()]
        m.append(x_return)
    return m


def predict2tag(predictions, sets):
    m = []
    for x in predictions:
        x_return = sets[x > 0.5]
        if len(x_return) == 0:
            x_return = sets[x == x.max()]
        m.append(x_return)
    return m


def predict1hot(predictions):
    m = []
    for x in predictions:
        x_return = (x > 0.5)*1
        if 1 not in x_return:
            x_return = (x == x.max())*1
        m.append(x_return)
    return np.array(m)