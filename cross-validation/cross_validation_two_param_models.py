# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 23:32:58 2018

@author: Georgios
"""

import numpy as np
import sys
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
sys.path.append('auxiliary/')
from data_processing import load_data, shape_data
from kfold_crosseval import kfold_crosseval

# settings
# methods: 'log-reg', 'svm-linear', 'svm-rbf', 'decision-tree', 'random-forest',
# 'naive-bayes', 'gradient-boosting', 'ada', 'ada2', 'knn',
# 'discriminant-analysis'
level = 'match'
norm = True
shuffle = True
method = 'ada'
min_round = 5
nsplits = 5

print('norm: %r - shuffle: %r - method: %s' % (norm, shuffle, method))

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
X_train, y_train, df, groups = shape_data(df, norm=norm, min_round=min_round)

print('Number of feaures:', X_train.shape[1], feats)
print('Number of obs:', X_train.shape[0])

# %% Set parameters
if method == 'svm-rbf':
    params1 = np.sort(np.concatenate((np.logspace(-5, 8, 14),
                                     5 * np.logspace(-5, 8, 14)), axis=0))
    params2 = np.sort(np.concatenate((np.logspace(-5, 8, 14),
                                     5 * np.logspace(-5, 8, 14)), axis=0))
elif method == 'ada':
    params1 = np.arange(5, 200, 1)
    params2 = np.arange(0.3, 1.5, 0.1)
else:
    sys.exit('Method not recognised')

# %% Tune parameters
accuracy = np.zeros((params1.shape[0], params2.shape[0]))
w_accuracy = np.zeros((params1.shape[0], params2.shape[0]))

for i, param1 in enumerate(params1):
    for j, param2 in enumerate(params2):

        if method == 'svm-rbf':
            model = SVC(C=param1, kernel='rbf', gamma=param2,
                        class_weight='balanced')
        elif method == 'ada':
            model = AdaBoostClassifier(n_estimators=param1, random_state=10,
                                       learning_rate=param2)

        # apply k-fold cross validation
        accuracy[i, j], w_accuracy[i, j] = kfold_crosseval(X_train, y_train,
                                                           df, nsplits,
                                                           groups=groups,
                                                           model=model,
                                                           level=level,
                                                           shuffle=shuffle)

np.savez('../output/%s' % method, accuracy=accuracy, w_accuracy=w_accuracy,
         params1=params1, params2=params2)

s = np.load('../output/%s.npz' % method)
accuracy = s['accuracy']
w_accuracy = s['w_accuracy']
plt.figure()
plt.imshow(accuracy)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(w_accuracy)
plt.colorbar()
plt.show()
