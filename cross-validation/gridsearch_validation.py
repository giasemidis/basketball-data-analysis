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
sys.path.append('auxiliary/')
from data_processing import load_data, shape_data


# settings
# methods: 'log-reg', 'svm-linear', 'svm-rbf', 'decision-tree', 'random-forest',
# 'naive-bayes', 'gradient-boosting', 'ada', 'ada2', 'knn',
# 'discriminant-analysis'
level = 'match'
norm = True
shuffle = True
method = 'ada2'
min_round = 5
nsplits = 5

print('level: %s - norm: %r - shuffle: %r - method: %s' %
      (level, norm, shuffle, method))

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
    feats = ['Home', 'Away', 'Position', 'Offence', 'Defence',
             'form', 'F4', 'Diff']

# seasons for calibration
df = df[df['Season'] < 2019]

# %% Re-shape data
X_train, y_train, df, groups = shape_data(df, feats, norm=norm,
                                          min_round=min_round)
print('Number of feaures:', X_train.shape[1], feats)
print('Number of obs:', X_train.shape[0])

if level == 'team':
    kfold = GroupKFold(n_splits=nsplits)
    folditer = kfold.split(X_train, y_train, groups)
else:
    kfold = StratifiedKFold(n_splits=nsplits, shuffle=shuffle, random_state=10)
    folditer = kfold.split(X_train, y_train)

# %% Set parameters
if method == 'log-reg':
    params = {'C': np.sort(np.concatenate((np.logspace(-5, 8, 14),
                           5 * np.logspace(-5, 8, 14)), axis=0))}
    model = LogisticRegression(solver='liblinear', class_weight='balanced')
elif method == 'svm-linear':
    params = {'C': np.sort(np.concatenate(
        (np.logspace(-5, 8, 14), 5 * np.logspace(-5, 8, 14)), axis=0))}
    model = SVC(kernel='linear', class_weight='balanced', random_state=10)
elif method == 'svm-rbf':
    params = {'C': np.sort(np.concatenate((np.logspace(-5, 6, 12),
                           5 * np.logspace(-5, 6, 12)), axis=0)),
              'gamma': np.sort(np.concatenate((np.logspace(-5, 6, 12),
                               5 * np.logspace(-5, 6, 12)), axis=0))}
    model = SVC(class_weight='balanced', random_state=10)
elif method == 'decision-tree':
    params = {}
    model = DecisionTreeClassifier(class_weight='balanced', random_state=10)
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
    params = {'n_estimators': np.arange(5, 200, 1)}
    model = AdaBoostClassifier(random_state=10, learning_rate=1)
elif method == 'ada2':
    params = {'n_estimators': np.arange(5, 200, 1),
              'learning_rate': np.concatenate(([0.01, 0.05],
                                               np.arange(0.1, 2.1, 0.1)))}
    model = AdaBoostClassifier(random_state=10)
elif method == 'ada3':
    params = {'n_estimators': np.arange(5, 200, 2),
              'learning_rate': np.arange(0.2, 2.1, 0.2),
              'base_estimator': [DecisionTreeClassifier(max_depth=1),
                                 DecisionTreeClassifier(max_depth=5),
                                 DecisionTreeClassifier(max_depth=10),
                                 DecisionTreeClassifier(max_depth=15),
                                 DecisionTreeClassifier(max_depth=20),
                                 DecisionTreeClassifier(max_depth=25),
                                 DecisionTreeClassifier(max_depth=30)]}
    model = AdaBoostClassifier(random_state=10)
elif method == 'knn':
    params = {'n_neighbors': np.arange(3, 50, 2)}
    model = KNeighborsClassifier()
elif method == 'discriminant-analysis':
    params = {}
    model = QuadraticDiscriminantAnalysis()
else:
    sys.exit('Method not recognised')

# %% Tune parameters

clf = GridSearchCV(model, params, cv=folditer, verbose=1, iid=False,
                   scoring=['accuracy', 'balanced_accuracy', 'roc_auc'],
                   refit='accuracy', n_jobs=-1)
clf.fit(X_train, y_train)

if hasattr(clf.best_estimator_, 'feature_importances_'):
    imp = clf.best_estimator_.feature_importances_
    ii = np.argsort(imp)[::-1]
    print('Feature Importance')
    print([(feats[u], imp[u]) for u in ii])

# %% Plots
accuracy = clf.cv_results_['mean_test_accuracy']
w_accuracy = clf.cv_results_['mean_test_balanced_accuracy']
roc_auc = clf.cv_results_['mean_test_roc_auc']
if len(params.keys()) == 0:
    print('Accuracy: ', accuracy[0])
    print('Weighted Accuracy: ', w_accuracy[0])
    print('ROC-AUC: ', roc_auc[0])
elif len(params.keys()) == 1:
    tmp = list(clf.param_grid)
    params = clf.param_grid[tmp[0]]
    print('Accuracy: %.4f at %.4g' %
          (np.max(accuracy), params[np.argmax(accuracy)]))
    print('Weighted Accuracy: %.4f at %.4g' %
          (np.max(w_accuracy), params[np.argmax(w_accuracy)]))
    print('ROC-AUC: %.4f at %.4g' %
          (np.max(roc_auc), params[np.argmax(roc_auc)]))
    plt.figure()
    plt.plot(params, accuracy, label='accuracy')
    plt.plot(params, w_accuracy, label='w_accuracy')
    plt.plot(params, roc_auc, label='ROC-AUC')
    if method in ['log-reg', 'svm-linear']:
        plt.xscale('log')
    plt.xlabel('parameter')
    plt.ylabel('Score')
    plt.legend()
    plt.title(method)
    plt.show()
elif len(params.keys()) == 2:

    # according to some references GridSearchCV() performs search in
    # alphabetical order of the parameters.
    tmp = sorted(list(clf.param_grid.keys()))
    shape = (clf.param_grid[tmp[0]].shape[0],
             clf.param_grid[tmp[1]].shape[0])
    accuracy = accuracy.reshape(shape)
    w_accuracy = w_accuracy.reshape(shape)
    np.savez('output/%s' % method, accuracy=accuracy,
             w_accuracy=w_accuracy,
             params1=clf.param_grid[tmp[0]], params2=clf.param_grid[tmp[1]])

    print('Accuracy: %.4f at %s=%.4g and %s=%.4g' %
          (clf.best_score_, tmp[0], clf.best_params_[tmp[0]],
           tmp[1], clf.best_params_[tmp[1]]))

    inds = np.unravel_index(np.argmax(accuracy), shape)
    print('Accuracy: %.4f at %s=%.4g and %s=%.4g' %
          (np.max(accuracy),
           tmp[0], clf.param_grid[tmp[0]][inds[0]],
           tmp[1], clf.param_grid[tmp[1]][inds[1]]))

    inds = np.unravel_index(np.argmax(w_accuracy), shape)
    print('Weighted Accuracy: %.4f at %s=%.4g and %s=%.4g' %
          (np.max(w_accuracy),
           tmp[0], clf.param_grid[tmp[0]][inds[0]],
           tmp[1], clf.param_grid[tmp[1]][inds[1]]))

    print('ROC-AUC: %.4f' % np.max(roc_auc))

    plt.figure()
    plt.imshow(accuracy)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(w_accuracy)
    plt.colorbar()
    plt.show()
elif len(params.keys()) == 3:
    print('Accuracy: %.4f at %s' % (clf.best_score_, clf.best_estimator_))
    np.savez('output/%s' % method,
             accuracy=clf.cv_results_['mean_test_accuracy'],
             w_accuracy=clf.cv_results_['mean_test_balanced_accuracy'],
             params=clf.cv_results_['params'])
