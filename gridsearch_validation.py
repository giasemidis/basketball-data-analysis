# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 23:32:58 2018

@author: Georgios
"""

import numpy as np
import sys
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from auxiliary.data_processing import load_data, shape_data
from auxiliary.kfold_crosseval import kfold_crosseval

# settings
level = 'match'
norm = True
shuffle = True
method = 'decision-tree'
min_round = 5
nsplits = 5

print('level: %s - norm: %r - shuffle: %r - method: %s' % 
      (level, norm, shuffle, method))

#%% load data
df = load_data(level)

#%% Re-shape data
X_train, y_train, df, init_feat, n_feats, groups = shape_data(df, norm=norm, 
                                                        min_round=min_round)
print('Number of feaures:', X_train.shape[1], init_feat)
print('Number of obs:', X_train.shape[0])

#%%
if level == 'team':
    kfold = GroupKFold(n_splits=nsplits)
    folditer = kfold.split(X_train, y_train, groups)
else:
    kfold = StratifiedKFold(n_splits=nsplits, shuffle=shuffle, random_state=10)
    folditer = kfold.split(X_train, y_train)
        
#%% Set parameters    
if method == 'log-reg':
    params = {'C': np.sort(np.concatenate((np.logspace(-5, 8, 14),
                                     5*np.logspace(-5, 8, 14)), axis=0))}
    model = LogisticRegression(solver='liblinear', class_weight='balanced')
elif method == 'svm-linear':
    params = {'C': np.sort(np.concatenate((np.logspace(-5, 8, 14),
                                     5*np.logspace(-5, 8, 14)), axis=0))}
    model = SVC(kernel='linear', class_weight='balanced')
elif method == 'decision-tree':
    params = {}
    model = DecisionTreeClassifier(random_state=10, class_weight='balanced')
elif method == 'random-forest':
    params = {'n_estimators': np.arange(10, 100, 5)}
    model = RandomForestClassifier(class_weight='balanced', random_state=10)
elif method == 'naive-bayes':
    params = {}
    model = GaussianNB()
elif method == 'gradient-boosting':
    params = {'n_estimators': np.arange(10, 200, 10)}
    model = GradientBoostingClassifier(random_state=10)
elif method == 'ada':
    params = {'n_estimators': np.arange(5, 200, 3)}
    model = AdaBoostClassifier(random_state=10, learning_rate=0.6)
elif method == 'knn':
    params = {'n_neighbors': np.arange(3, 30, 2)}
    model = KNeighborsClassifier()
elif method == 'discriminant-analysis': 
    params = {}
    model = QuadraticDiscriminantAnalysis()
else:
    sys.exit('Method not recognised')

#%% Tune parameters

clf = GridSearchCV(model, params, cv=folditer,
                   scoring=['accuracy', 'balanced_accuracy' 'roc_auc'],
                   refit='accuracy')
clf.fit(X_train, y_train)

#%% Plots

#if params.shape[0] > 1:
#    plt.figure()
#    plt.plot(params, accuracy, label='accuracy')
#    plt.plot(params, w_accuracy, label='w_accuracy')
#    if method in ['log-reg', 'svm-linear']:
#        plt.xscale('log')
#    plt.xlabel('parameter')
#    plt.ylabel('Score')
#    plt.legend()
#    plt.title(method)
#else:
#    print('Accuracy: ', accuracy.mean(axis=0))
#    print('Weighted Accuracy: ', w_accuracy.mean(axis=0))
