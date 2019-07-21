# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 23:32:58 2018

@author: Georgios
"""

import numpy as np
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
sys.path.append('auxiliary/')
from data_processing import load_data, shape_data
from kfold_crosseval import kfold_crosseval

import warnings
warnings.filterwarnings("ignore")
# settings
# methods: 'log-reg', 'svm-linear', 'svm-rbf', 'decision-tree', 'random-forest',
# 'naive-bayes', 'gradient-boosting', 'ada', 'ada2', 'knn',
# 'discriminant-analysis'
level = 'team'
norm = True
shuffle = True
method = 'svm-rbf'
min_round = 5
nsplits = 5

print('norm: %r - shuffle: %r - method: %s' % (norm, shuffle, method))

# %% load data
df = load_data(level)

# choose features
if level == 'match':
    # 'Round', 'Season', 'Home Team', 'Away Team', 'Label',
    feats = ['Position_x', 'Position_y', 'Offence_x', 'Offence_y',
             'Defence_x', 'Defence_y', 'form_x', 'form_y', 'Diff_x', 'Diff_y',
             'Home F4', 'Away F4']
elif level == 'team':
    # 'Round', 'Season', 'Game ID', 'Team', 'Label',
    feats = ['Home', 'Away', 'Position',
             'Offence', 'Defence', 'form', 'F4', 'Diff']

# seasons for calibration
df = df[df['Season'] < 2019]

# %% Re-shape data
X_train, y_train, df, groups = shape_data(df, feats, norm=norm,
                                          min_round=min_round)

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

for i, param1 in enumerate(tqdm(params1, desc='1st loop')):
    for j, param2 in enumerate(tqdm(params2, desc='2st loop')):

        if method == 'svm-rbf':
            model = SVC(C=param1, kernel='rbf', gamma=param2,
                        class_weight='balanced', probability=True,
                        max_iter=400)
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
    np.savez('output/%s' % method, accuracy=accuracy, w_accuracy=w_accuracy,
             params1=params1, params2=params2)

print('Accuracy: ', np.round(np.max(accuracy), 4))
print('Weighted Accuracy: ', np.round(np.max(w_accuracy), 4))

plt.imshow(accuracy)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(w_accuracy)
plt.colorbar()
plt.show()
