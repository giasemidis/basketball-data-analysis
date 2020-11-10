# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 22:51:26 2019

@author: giase
"""
import numpy as np
import sys
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
sys.path.append('auxiliary/')
from data_processing import load_data, shape_data

level = 'match'
min_round = 5
norm = False

df = load_data(level)

params = [
#          {'features': ['Position_x', 'Position_y', 'Offence_x', 'Defence_x',
#                        'Defence_y', 'form_x', 'form_y', 'Diff_y'], 
#           'n_estimators': 117, 'learning_rate': 1.1},
#          {'features': ['Position_x', 'Position_y', 'Offence_x', 'Defence_x',
#                        'Defence_y', 'form_x', 'form_y', 'Diff_y', 'Home F4'],
#           'n_estimators': 71, 'learning_rate': 1.1},
#          {'features': ['Position_x', 'Position_y', 'Offence_x', 'Defence_x',
#                        'Defence_y', 'form_x', 'form_y', 'Diff_y', 'Home F4'],
#           'n_estimators': 117, 'learning_rate': 1.1},
#          {'features': ['Position_x', 'Offence_x', 'Offence_y', 'Defence_y',
#                        'Diff_y', 'Home F4', 'Away F4'],
#           'n_estimators': 28, 'learning_rate': 1.0},
          {'features': ['Position_x', 'Position_y', 'Offence_x', 'Offence_y',
             'Defence_x', 'Defence_y',
             'form_x', 'form_y',
             'Diff_x', 'Diff_y',
             'Home F4', 'Away F4'],
           'n_estimators': 130, 'learning_rate': 0.7},
#          {'features': ['Position_x', 'Position_y', 'Offence_x', 'Offence_y',
#                        'Defence_y', 'Diff_y', 'Away F4'],
#           'n_estimators': 128, 'learning_rate': 1.0}
]
# 128 - 0.8
# 130 - 0.05
# %%
#base = DecisionTreeClassifier(max_depth=25)

for param in tqdm(params):
    features = param['features']
    n_estimators = param['n_estimators']
    learning_rate = param['learning_rate']
    model = AdaBoostClassifier(n_estimators=n_estimators, random_state=10,
                               learning_rate=learning_rate)

    X_train, y_train, _, _ = shape_data(df[df['Season'] != 2019], features,
                                        norm=norm, min_round=min_round)
    X_test, y_test, _, _ = shape_data(df[df['Season'] == 2019], features, 
                                      norm=norm, min_round=min_round)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    waccuracy = balanced_accuracy_score(y_test, y_pred)
    tqdm.write('Accuracy: %.4f - W-Accuracy: %.4f' % (accuracy, waccuracy))
    