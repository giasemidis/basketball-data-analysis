# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 22:35:09 2018

@author: Georgios
"""

import numpy as np


def sample_weights(y, n_classes):
    ns = np.bincount(y, minlength=n_classes)
    nonzeroinde = ns!=0
    w = np.zeros(n_classes)
    w[nonzeroinde] = y.shape[0] / (np.unique(y).shape[0] * ns[nonzeroinde])
    weights = np.zeros(y.shape[0])
    for i in range(n_classes):
        weights[y==i] = w[i]

    return weights