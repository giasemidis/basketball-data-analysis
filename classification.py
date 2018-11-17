# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 23:32:58 2018

@author: Georgios
"""

import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from auxiliary.sample_weights import sample_weights

df = pd.read_csv('data/match_level_features.csv')
df2 = pd.read_csv('data/euroleague_season_2018_results-end.csv')

yy = df2['Home Team ID']

y = df['label'].values
X = df.iloc[:, 2:].values

#ii = X.sum(axis=1) == -X.shape[1]
#y = y[~ii]
#X = X[~ii, :]
#yy = yy[~ii]

kk = 40
y = y[kk:]
X = X[kk:, :]
yy = yy [kk:]

method = 'svm-linear'

if method == 'log-reg':
    params = np.sort(np.concatenate((np.logspace(-5, 8, 14),
                                     5*np.logspace(-5, 8, 14)), axis=0))
elif method == 'svm-linear':
    params = np.sort(np.concatenate((np.logspace(-5, 8, 14),
                                     5*np.logspace(-5, 8, 14)), axis=0))
elif method == 'decision-tree':
    params = np.array([0])
elif method == 'random-forest':
    params = np.arange(10, 100, 5)
elif method == 'naive-bayes':
    params = np.array([0])
elif method == 'gradient-boosting':
    params = np.arange(10, 200, 10)
else:
    sys.exit('Method not recognised')

skfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=10)

accuracy = np.zeros((skfold.get_n_splits(), params.shape[0]))
w_accuracy = np.zeros((skfold.get_n_splits(), params.shape[0]))

i = -1
for train_index, test_index in skfold.split(X, yy):
    i+=1
    print(i)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    w = sample_weights(y_test-1, 2)
    
    for j, param in enumerate(params):
    
        if method == 'log-reg':
            model = LogisticRegression(C=param, class_weight='balanced')
        elif method == 'svm-linear':
            model = SVC(C=param, kernel='linear')
        elif method == 'decision-tree':
            model = DecisionTreeClassifier()
        elif method == 'random-forest':
            model = RandomForestClassifier(n_estimators=param)
        elif method == 'naive-bayes':
            model = GaussianNB()
        elif method == 'gradient-boosting':
            model = GradientBoostingClassifier(n_estimators=param)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy[i, j] = accuracy_score(y_test, y_pred)
        w_accuracy[i, j] = accuracy_score(y_test, y_pred, sample_weight=w)

if params.shape[0] > 1:
    plt.figure()
    plt.plot(params, accuracy.mean(axis=0), label='accuracy')
    plt.plot(params, w_accuracy.mean(axis=0), label='w_accuracy')
    if method in ['log-reg', 'svm-linear']:
        plt.xscale('log')
    plt.xlabel('parameter')
    plt.ylabel('Score')
    plt.legend()
else:
    print('Accuracy: ', accuracy.mean(axis=0))
    print('Weighted Accuracy: ', w_accuracy.mean(axis=0))