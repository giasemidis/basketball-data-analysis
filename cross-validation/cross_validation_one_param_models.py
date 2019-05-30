# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 23:32:58 2018

@author: Georgios
"""

import numpy as np
import sys
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
sys.path.append('auxiliary/')
from data_processing import load_data, shape_data
from kfold_crosseval import kfold_crosseval

# settings
# methods: 'log-reg', 'svm-linear', 'svm-rbf', 'decision-tree', 'random-forest',
# 'naive-bayes', 'gradient-boosting', 'ada', 'ada2', 'knn',
# 'discriminant-analysis'
level = 'team'
norm = True
shuffle = True
method = 'log-reg'
min_round = 5
nsplits = 5

print('level: %s - norm: %r - shuffle: %r - method: %s' %
      (level, norm, shuffle, method))

# %% load data
df = load_data(level)

# choose features
if level == 'match':
    feats = ['Round', 'Season', 'Home Team', 'Away Team', 'Label',
             'Position_x', 'Position_y', 'Offence_x', 'Offence_y',
             'Defence_x', 'Defence_y', 'form_x', 'form_y', 'Diff_x', 'Diff_y',
             'Home F4', 'Away F4']
elif level == 'team':
    feats = ['Round', 'Season', 'Game ID', 'Team', 'Label',
             'Home', 'Away', 'Position',
             'Offence', 'Defence', 'form', 'F4', 'Diff']

# seasons for calibration
df = df[df['Season'] < 2019]

# %% Re-shape data
X_train, y_train, df, groups = shape_data(df, feats, norm=norm,
                                          min_round=min_round)
print('Number of feaures:', X_train.shape[1], feats)
print('Number of obs:', X_train.shape[0])

# %% Set parameters
if method == 'log-reg':
    params = np.sort(np.concatenate((np.logspace(-5, 8, 14),
                                     5 * np.logspace(-5, 8, 14)), axis=0))
elif method == 'svm-linear':
    params = np.sort(np.concatenate((np.logspace(-5, 6, 12),
                                     5 * np.logspace(-5, 6, 12)), axis=0))
elif method == 'decision-tree':
    params = np.array([0])
elif method == 'random-forest':
    params = np.arange(10, 100, 5)
elif method == 'naive-bayes':
    params = np.array([0])
elif method == 'gradient-boosting':
    params = np.arange(10, 200, 10)
elif method == 'ada':
    params = np.arange(5, 200, 3)
elif method == 'knn':
    params = np.arange(3, 30, 2)
elif method == 'discriminant-analysis':
    params = np.array([0])
else:
    sys.exit('Method not recognised')

# %% Tune parameters
accuracy = np.zeros(params.shape[0])
w_accuracy = np.zeros(params.shape[0])

for j, param in enumerate(params):

    # update model's parameters
    if method == 'log-reg':
        model = LogisticRegression(C=param, solver='liblinear',
                                   class_weight='balanced')
    elif method == 'svm-linear':
        model = SVC(C=param, kernel='linear', class_weight='balanced',
                    probability=True)
    elif method == 'decision-tree':
        model = DecisionTreeClassifier(class_weight='balanced', random_state=10)
    elif method == 'random-forest':
        model = RandomForestClassifier(n_estimators=param,
                                       class_weight='balanced',
                                       random_state=10)
    elif method == 'naive-bayes':
        model = GaussianNB()
    elif method == 'gradient-boosting':
        model = GradientBoostingClassifier(n_estimators=param, random_state=10)
    elif method == 'ada':
        model = AdaBoostClassifier(n_estimators=param, random_state=10,
                                   learning_rate=0.6)
    elif method == 'knn':
        model = KNeighborsClassifier(n_neighbors=param)
    elif method == 'discriminant-analysis':
        model = QuadraticDiscriminantAnalysis()
    else:
        sys.exit('method name is not valid')

    # apply k-fold cross validation
    accuracy[j], w_accuracy[j] = kfold_crosseval(X_train, y_train,
                                                 df, nsplits, groups=groups,
                                                 model=model, level=level,
                                                 shuffle=shuffle)

# %% Plots
if params.shape[0] > 1:
    print('Accuracy: ', np.round(np.max(accuracy), 4))
    print('Weighted Accuracy: ', np.round(np.max(w_accuracy), 4))
    plt.figure()
    plt.plot(params, accuracy, label='accuracy')
    plt.plot(params, w_accuracy, label='w_accuracy')
    if method in ['log-reg', 'svm-linear']:
        plt.xscale('log')
    plt.xlabel('parameter')
    plt.ylabel('Score')
    plt.legend()
    plt.title(method)
    plt.show()
else:
    print('Accuracy: ', accuracy.mean(axis=0))
    print('Weighted Accuracy: ', w_accuracy.mean(axis=0))
