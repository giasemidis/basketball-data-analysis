# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:25:38 2019

@author: Georgios
"""
import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from auxiliary.data_processing import load_data, shape_data
from auxiliary.kfold_crosseval import kfold_crosseval

level = 'match'
shuffle = True
norm = True
min_round = 5
nsplits = 5
#model = GaussianNB()
model= AdaBoostClassifier(n_estimators=60, random_state=10,
                          learning_rate=0.5)

#%% load data
df = load_data(level)

#%% Re-shape data
X_train, y_train, df, init_feat, n_feats, groups = shape_data(df, norm=norm, min_round=min_round)

#%% Apply PCA and then k-fold cross validation

XX = X_train.copy()

scores = np.zeros((n_feats, 2))
for n in range(n_feats):
    pca = PCA(n_components=n+1)
    X_train = pca.fit_transform(XX)
    scores[n, 0], scores[n, 1] = kfold_crosseval(X_train, y_train, df,
                                                 nsplits, groups=groups, 
                                                 model=model, 
                                                 level=level, 
                                                 shuffle=shuffle)

#%% Plots
x = np.arange(1, n_feats+1, dtype=int)
plt.figure()
plt.plot(x, scores[:, 0], label='Accuracy')
plt.plot(x, scores[:, 1], label='W-Accuracy')
plt.xticks(x, x)
plt.grid()
plt.legend()