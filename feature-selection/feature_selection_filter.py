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

from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot as plt
sys.path.append('auxiliary/')
from data_processing import load_data, shape_data
from kfold_crosseval import kfold_crosseval

level = 'match'
shuffle = True
norm = True
min_round = 5
nsplits = 5
# mode = GaussianNB()
model = AdaBoostClassifier(n_estimators=115, random_state=10,
                           learning_rate=1.1)

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

# %% Filter based feature selection (model independent)
fscores, ps = f_classif(X_train, y_train)
mscores = mutual_info_classif(X_train, y_train)
chiscores = chi2(X_train, y_train)
ordered_fc = [feats[u] for u in np.argsort(fscores)[::-1]]
ordered_mi = [feats[u] for u in np.argsort(mscores)[::-1]]
ordered_ch = [feats[u] for u in np.argsort(chiscores)[::-1]]
print('F scores', ordered_fc)
print('MI scores', ordered_mi)
print('Chi scores', ordered_ch)

# %%
accuracy = np.zeros((n_feats, 2))
w_accuracy = np.zeros((n_feats, 2))
feats_fs = []
feats_mi = []
for i, n in enumerate(range(1, n_feats + 1)):
    kk = n if n < n_feats else 'all'
    skb_fc = SelectKBest(f_classif, k=kk)
    skb_mi = SelectKBest(mutual_info_classif, k=kk)
    skb_ch = SelectKBest(chi2, k=kk)
    X_fc = skb_fc.fit_transform(X_train, y_train)
    X_mi = skb_mi.fit_transform(X_train, y_train)
    X_ch = skb_ch.fit_transform(X_train, y_train)

    accuracy[i, 0], w_accuracy[i, 0] = kfold_crosseval(X_fc, y_train, df,
                                                       nsplits, groups=groups,
                                                       model=model,
                                                       level=level,
                                                       shuffle=shuffle)
    accuracy[i, 1], w_accuracy[i, 1] = kfold_crosseval(X_mi, y_train, df,
                                                       nsplits, groups=groups,
                                                       model=model,
                                                       level=level,
                                                       shuffle=shuffle)
#    feats_fs.append(ordered_fc[:n])
#    feats_mi.append(ordered_mi[:n])
# %%
x = np.arange(1, n_feats + 1)
plt.figure()
plt.plot(x, accuracy[:, 0], label='Accuracy')
plt.plot(x, w_accuracy[:, 0], label='W-Accuracy')
plt.xticks(x, x)
plt.minorticks_on()
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle='--')
plt.title('ANOVA')
plt.legend()
plt.show()

plt.figure()
plt.plot(x, accuracy[:, 1], label='Accuracy')
plt.plot(x, w_accuracy[:, 1], label='W-Accuracy')
plt.ylabel('Score')
plt.xticks(x, x)
plt.minorticks_on()
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle='--')
plt.title('MI')
plt.legend()
plt.show()
