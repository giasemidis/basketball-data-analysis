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

norm = False
shuffle = False
merge = True
method = 'svm-rbf'
min_round = 5

print('norm: %r - shuffle: %r - merge: %r - method: %s' % 
      (norm, shuffle, merge, method))

#df1 = pd.read_csv('data/match_level_features_2016_2017.csv')
#df2 = pd.read_csv('data/match_level_features_2017_2018.csv')

df1 = pd.read_csv('data/team_level_features_2016_2017.csv')
df2 = pd.read_csv('data/team_level_features_2017_2018.csv')

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
    
if method == 'svm-rbf':
    params = np.sort(np.concatenate((np.logspace(-5, 8, 14),
                                     5*np.logspace(-5, 8, 14)), axis=0))
    gammas = np.sort(np.concatenate((np.logspace(-5, 8, 14),
                                     5*np.logspace(-5, 8, 14)), axis=0))
else:
    sys.exit('Method not recognised')

skfold = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=10)

accuracy = np.zeros((skfold.get_n_splits(), params.shape[0], gammas.shape[0]))
w_accuracy = np.zeros((skfold.get_n_splits(), params.shape[0], gammas.shape[0]))

i = -1
for train_index, test_index in skfold.split(X_train, y_train):
    i+=1
    print(i)
    X_train_folds, X_test_fold = X_train[train_index,:], X_train[test_index,:]
    y_train_folds, y_test_fold = y_train[train_index], y_train[test_index]
    w = sample_weights(y_test_fold, 2)
    
    for j, param in enumerate(params):
        for k, g in enumerate(gammas):
            
            if method == 'svm-rbf':
                model = SVC(C=param, kernel='rbf', gamma=g,
                            class_weight='balanced')

                model.fit(X_train_folds, y_train_folds)
                y_pred = model.predict(X_test_fold)
                accuracy[i, j, k] = accuracy_score(y_test_fold, y_pred)
                w_accuracy[i, j, k] = accuracy_score(y_test_fold, y_pred,
                                                     sample_weight=w)

mean_acc = np.mean(accuracy, axis=0)
mean_w_acc = np.mean(w_accuracy, axis=0)

np.savez('svm-rbf', accuracy=accuracy, w_accuracy=w_accuracy, Cs=params,
         gammas=gammas)
#%%
plt.figure()
plt.imshow(mean_acc)
plt.colorbar()

plt.figure()
plt.imshow(mean_w_acc)
plt.colorbar()

