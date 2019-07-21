# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:25:38 2019

@author: Georgios
"""
import numpy as np
import sys
from tqdm import tqdm
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


def plot(x, accuracy, w_accuracy, title=''):
    plt.figure()
    plt.plot(x, accuracy, label='Accuracy')
    plt.plot(x, w_accuracy, label='W-Accuracy')
    plt.xlabel('Number of features')
    plt.ylabel('Score')
    plt.xticks(x, x)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-')
    plt.grid(which='minor', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.show()
    return


def mutual_info_classif2(X, y, discrete_features='auto', n_neighbors=3,
                         copy=True, random_state=10):
    return mutual_info_classif(X, y, discrete_features=discrete_features,
                               n_neighbors=n_neighbors, copy=copy,
                               random_state=random_state)


level = 'match'
shuffle = True
norm = True
min_round = 5
nsplits = 5
# mode = GaussianNB()
random_state = 10
model = AdaBoostClassifier(n_estimators=121, random_state=random_state,
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

# %% Filter based feature selection (model independent)
fscores, ps = f_classif(X_train, y_train)
mscores = mutual_info_classif(X_train, y_train, random_state=10)
chiscores, _ = chi2(X_train, y_train)
ordered_fc = [feats[u] for u in np.argsort(fscores)[::-1]]
ordered_mi = [feats[u] for u in np.argsort(mscores)[::-1]]
ordered_ch = [feats[u] for u in np.argsort(chiscores)[::-1]]
print('F scores', ordered_fc)
print('MI scores', ordered_mi)
print('Chi scores', ordered_ch)

# %%
accuracy = np.zeros((n_feats, 3))
w_accuracy = np.zeros((n_feats, 3))
feats_fs = []
feats_mi = []
for i, n in enumerate(tqdm(range(1, n_feats + 1))):
    kk = n if n < n_feats else 'all'
    skb_fc = SelectKBest(f_classif, k=kk)
    skb_mi = SelectKBest(mutual_info_classif2, k=kk)
    skb_ch = SelectKBest(chi2, k=kk)
    X_fc = skb_fc.fit_transform(X_train, y_train)
    X_mi = skb_mi.fit_transform(X_train, y_train)
    X_ch = skb_ch.fit_transform(X_train, y_train)

    # print('MI:', skb_mi.scores_)
    # print(skb_mi.get_support())

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
    accuracy[i, 2], w_accuracy[i, 2] = kfold_crosseval(X_ch, y_train, df,
                                                       nsplits, groups=groups,
                                                       model=model,
                                                       level=level,
                                                       shuffle=shuffle)

#    feats_fs.append(ordered_fc[:n])
#    feats_mi.append(ordered_mi[:n])
# %%

x = np.arange(1, n_feats + 1)
plot(x, accuracy[:, 0], w_accuracy[:, 0], title='ANOVA')
plot(x, accuracy[:, 1], w_accuracy[:, 1], title='MI')
plot(x, accuracy[:, 2], w_accuracy[:, 2], title='Chi2')

scores = np.concatenate((fscores[:, None], mscores[:, None],
                         chiscores[:, None]), axis=1)
order = np.argsort(scores, axis=0)
ranks = order.argsort(axis=0)

# %%
plt.figure()
plt.imshow((scores.shape[0] - ranks).T)
plt.yticks(ticks=[0, 1, 2], labels=['ANOVA', 'MI', 'Chi2'])
plt.xticks(ticks=np.arange(len(feats)), labels=feats, rotation='vertical')
plt.colorbar(orientation='horizontal', pad=0.3)
plt.show()
