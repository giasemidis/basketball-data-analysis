# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:25:38 2019

@author: Georgios
"""
import numpy as np
import sys
from tqdm import tqdm
from itertools import combinations
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
sys.path.append('auxiliary/')
from data_processing import load_data, shape_data
from kfold_crosseval import kfold_crosseval

level = 'match'
shuffle = True
norm = True
min_round = 5
nsplits = 5
# model = GaussianNB()
model = AdaBoostClassifier(n_estimators=121, random_state=10,
                           learning_rate=1.0)

# %% load data
df = load_data(level)

# choose features
if level == 'match':
    # 'Round', 'Season', 'Home Team', 'Away Team', 'Label',
    feats = ['Position_x', 'Position_y', 'Offence_x', 'Offence_y',
             'Defence_x', 'Defence_y',
             'form_x', 'form_y',
             'Diff_x', 'Diff_y',
             'Home F4', 'Away F4']
elif level == 'team':
    # 'Round', 'Season', 'Game ID', 'Team', 'Label',
    feats = ['Home', 'Away', 'Position', 'Offence', 'Defence',
             'form',
             'F4', 'Diff']
n_feats = len(feats)

# seasons for calibration
df = df[df['Season'] < 2019]

# %% Re-shape data
X_train, y_train, df, groups = shape_data(df, feats, norm=norm,
                                          min_round=min_round)

# %% Embedded feature selection (combinations of features)

# create all possible combination of features.
allcombs = []
for u in range(1, n_feats + 1):
    combs = combinations(feats, u)
    for c in combs:
        if list(c) != []:
            allcombs.append(list(c))

scores = np.zeros((len(allcombs), 2))
nc = 0
for ii, comb in enumerate(tqdm(allcombs)):

    if len(comb) > nc:
        tqdm.write('Number of features: %d' % len(comb))
        nc = len(comb)
    indx, feats = [], []

    X_train = df[comb].values

    scores[ii, 0], scores[ii, 1] = kfold_crosseval(X_train, y_train, df,
                                                   nsplits, groups=groups,
                                                   model=model,
                                                   level=level,
                                                   shuffle=shuffle)

np.savez('output/wrapper', scores=scores, features=np.array(allcombs))
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
