# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:25:38 2019

@author: Georgios
"""
import numpy as np
import sys
from itertools import combinations
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

# create all possible combination of features.
allcombs = []
for u in range(1, n_feats + 1):
    combs = combinations(init_feat, u)
    for c in combs:
        if list(c) != []:
            allcombs.append(list(c))

scores = np.zeros((len(allcombs), 2))

for ii, comb in enumerate(allcombs):
    print('Number of features:', len(comb))
    indx, feats = [], []

    X_train = df[comb].values

    scores[ii, 0], scores[ii, 1] = kfold_crosseval(X_train, y_train, df,
                                                   nsplits, groups=groups,
                                                   model=model,
                                                   level=level,
                                                   shuffle=shuffle)

# Sort best combinations
ll = np.argsort(scores[:, 0])[::-1]
sortcombs = [allcombs[u] for u in ll]

x = np.arange(1, len(allcombs) + 1, dtype=int)
plt.figure()
plt.plot(x, scores[:, 0], label='Accuracy')
plt.plot(x, scores[:, 1], label='W-Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.bar(x[:15], scores[ll, 0][:15])
plt.xticks(x[:15], sortcombs[:15], rotation='vertical')
plt.show()
