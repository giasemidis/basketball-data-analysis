# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 23:32:58 2018

@author: Georgios
"""

import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib import pyplot as plt
from auxiliary.data_processing import load_data, shape_data
from auxiliary.kfold_crosseval import kfold_crosseval

# settings
level = 'match'
norm = True
shuffle = True
merge = True
year = 2017
method = 'naive-bayes'
min_round = 5
nsplits = 5

print('level: %s - norm: %r - shuffle: %r - merge: %r - method: %s' % 
      (level, norm, shuffle, merge, method))

#%% load data
df = load_data(level)

#%% Re-shape data
X_train, y_train, df, init_feat, n_feats, groups = shape_data(df, norm=norm, 
                                                        min_round=min_round)
print('Number of feaures:', X_train.shape[1], init_feat)
print('Number of obs:', X_train.shape[0])

#%% Set parameters    
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
elif method == 'ada':
    params = np.arange(10, 200, 10)
elif method == 'knn':
    params = np.arange(3, 30, 2)
elif method == 'distriminant-analysis': 
    params = np.array([0])
else:
    sys.exit('Method not recognised')

#%% Tune parameters
accuracy = np.zeros(params.shape[0])
w_accuracy = np.zeros(params.shape[0])

for j, param in enumerate(params):
     
    # update model's parameters
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
    elif method == 'ada':
        model = AdaBoostClassifier(n_estimators=param)
    elif method == 'knn':
        model = KNeighborsClassifier(n_neighbors=param)
    elif method == 'distriminant-analysis': 
        model = QuadraticDiscriminantAnalysis()

    # apply k-fold cross validation
    accuracy[j], w_accuracy[j] = kfold_crosseval(X_train, y_train, 
                                                 df, nsplits, groups=groups, 
                                                 model=model, level=level, 
                                                 shuffle=shuffle)

#%% Plots
if params.shape[0] > 1:
    plt.figure()
    plt.plot(params, accuracy, label='accuracy')
    plt.plot(params, w_accuracy, label='w_accuracy')
    if method in ['log-reg', 'svm-linear']:
        plt.xscale('log')
    plt.xlabel('parameter')
    plt.ylabel('Score')
    plt.legend()
    plt.title(method)
else:
    print('Accuracy: ', accuracy.mean(axis=0))
    print('Weighted Accuracy: ', w_accuracy.mean(axis=0))
