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
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
sys.path.append('auxiliary/')
from data_processing import load_data, shape_data
# from auxiliary.kfold_crosseval import kfold_crosseval

level = 'match'
shuffle = True
norm = True
min_round = 5
nsplits = 5
# model = GaussianNB()
# model = AdaBoostClassifier(n_estimators=115, random_state=10,
#                           learning_rate=1.1)

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

# %% Apply PCA and then k-fold cross validation

XX = X_train.copy()
params = {'n_estimators': np.arange(5, 200, 1),
          'learning_rate': np.arange(0.3, 1.5, 0.1)}
model = AdaBoostClassifier(random_state=10)
scores = np.zeros((n_feats, 2))
for n in range(n_feats):
    print(n)
    pca = PCA(n_components=n + 1)
    X_train = pca.fit_transform(XX)
#    scores[n, 0], scores[n, 1] = kfold_crosseval(X_train, y_train, df,
#                                                 nsplits, groups=groups,
#                                                 model=model,
#                                                 level=level,
#                                                 shuffle=shuffle)
    kfold = StratifiedKFold(n_splits=nsplits, shuffle=shuffle, random_state=10)
    folditer = kfold.split(X_train, y_train)
    clf = GridSearchCV(model, params, cv=folditer, iid=False,
                       scoring=['accuracy', 'balanced_accuracy', 'roc_auc'],
                       refit='accuracy')
    clf.fit(X_train, y_train)
    scores[n, 0] = np.max(clf.cv_results_['mean_test_accuracy'])
    scores[n, 1] = np.max(clf.cv_results_['mean_test_balanced_accuracy'])
    print(clf.best_score_)

# %% Plots
x = np.arange(1, n_feats + 1, dtype=int)
plt.figure()
plt.plot(x, scores[:, 0], label='Accuracy')
plt.plot(x, scores[:, 1], label='W-Accuracy')
plt.xticks(x, x)
plt.grid()
plt.legend()
plt.show()
