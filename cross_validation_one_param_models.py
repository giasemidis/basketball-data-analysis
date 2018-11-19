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
from auxiliary.fix_team_names import fix_team_names

def normalise(X):
    x_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return x_norm

norm = True
shuffle = False
merge = True
method = 'naive-bayes'
min_round = 5

print('norm: %r - shuffle: %r - merge: %r - method: %s' % 
      (norm, shuffle, merge, method))

df1 = pd.read_csv('data/match_level_features_2016_2017.csv')
df2 = pd.read_csv('data/match_level_features_2017_2018.csv')

#df1, df2 = fix_team_names(df1, df2)
#%%

n = list(df1.columns).index('Label')
ii = df1['Round'].values > min_round
y_train = df1[ii]['Label'].values
X_train = df1.iloc[ii, (n+1):].values

n = list(df2.columns).index('Label')
ii = df2['Round'].values > min_round
y_test = df2[ii]['Label'].values
X_test = df2.iloc[ii, (n+1):].values

if merge:
    X_train = np.concatenate((X_train, X_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

if norm:
    X_train = normalise(X_train)
    X_test = normalise(X_test)
    
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

skfold = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=10)

accuracy = np.zeros((skfold.get_n_splits(), params.shape[0]))
w_accuracy = np.zeros((skfold.get_n_splits(), params.shape[0]))

i = -1
for train_index, test_index in skfold.split(X_train, y_train):
    i+=1
    print(i)
    X_train_folds, X_test_fold = X_train[train_index, :], X_train[test_index, :]
    y_train_folds, y_test_fold = y_train[train_index], y_train[test_index]
    w = sample_weights(y_test_fold-1, 2)
    
    for j, param in enumerate(params):
    
        if method == 'log-reg':
            model = LogisticRegression(C=param, class_weight='balanced')
        elif method == 'svm-linear':
            model = SVC(C=param, kernel='linear', class_weight='balanced')
        elif method == 'decision-tree':
            model = DecisionTreeClassifier(class_weight='balanced')
        elif method == 'random-forest':
            model = RandomForestClassifier(n_estimators=param, class_weight='balanced')
        elif method == 'naive-bayes':
            model = GaussianNB()
        elif method == 'gradient-boosting':
            model = GradientBoostingClassifier(n_estimators=param)

        model.fit(X_train_folds, y_train_folds)
        y_pred = model.predict(X_test_fold)
        accuracy[i, j] = accuracy_score(y_test_fold, y_pred)
        w_accuracy[i, j] = accuracy_score(y_test_fold, y_pred, sample_weight=w)

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
