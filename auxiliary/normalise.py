# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 00:14:34 2019

@author: Georgios
"""


def normalise(X):
    x_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return x_norm
