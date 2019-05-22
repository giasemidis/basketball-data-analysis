# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:25:38 2019

@author: Georgios
"""
import numpy as np
import sys
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
sys.path.append('..')
from auxiliary.data_processing import load_data, shape_data
from auxiliary.kfold_crosseval import kfold_crosseval

level = 'match'
shuffle = True
norm = True
min_round = 5
nsplits = 5
# model = GaussianNB()
model = AdaBoostClassifier(n_estimators=115, random_state=10,
                           learning_rate=1.1)

# %% load data
df = load_data(level)

# %% Re-shape data
X_train, y_train, df, init_feat, n_feats, groups = shape_data(
    df, norm=norm, min_round=min_round)


# %% Embedded feature selection (combinations of features)
# add features one by one

# create a copy of the initial X_train.
XX = X_train.copy()
# create the indices of the features
allfeats = np.arange(n_feats, dtype=int)
# lists to keep the best features and their accuracies scores.
bestfeats = []
accuracy = []
w_accuracy = []
while len(allfeats) > 0:
    # number of remaining features
    n_temp_feat = len(allfeats)
    print('Number of features to process from:', n_temp_feat)
    # indices of current best features
    c_best = np.array(bestfeats.copy(), dtype=int)
    temp_acc = np.zeros(n_temp_feat)
    temp_wacc = np.zeros(n_temp_feat)
    for n in range(n_temp_feat):
        # append current best features with the features remaining in the list
        # (one by one)
        cfeat = np.append(c_best, allfeats[n])
        print('Indices of features under process:', cfeat)
        # select these features from the total design matrix.
        X_train = XX[:, cfeat]
        # run k-fold cross validation
        temp_acc[n], temp_wacc[n] = kfold_crosseval(X_train, y_train, df,
                                                    nsplits, groups=groups,
                                                    model=model,
                                                    level=level,
                                                    shuffle=shuffle)
    # find index of max accuracy
    nn = np.argmax(temp_acc)
    # append list of indices of best features with the index of the new best
    # feature
    bestfeats.append(allfeats[nn])
    # similarly for accuracy scores
    accuracy.append(temp_acc[nn])
    w_accuracy.append(temp_wacc[nn])
    allfeats = np.delete(allfeats, nn)
    print('Best Features:', bestfeats)

# %% Plots
x = np.arange(1, n_feats + 1)
plt.figure()
plt.plot(x, accuracy, label='Accuracy')
plt.plot(x, w_accuracy, label='W-Accuracy')
plt.xticks(x, x)
plt.minorticks_on()
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle='--')
# plt.tight_layout()
plt.show()
