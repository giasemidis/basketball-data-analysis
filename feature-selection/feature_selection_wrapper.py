'''
Wrapper method for feature selection, i.e. subsets of features are
generated and evaluated using a chosen algorithm and its hyper-parameters.

Here, as the number of features is relative small, we are able to generate
all possible combinations of features. If the number of features grows large,
a different approach should be adopted, the Sequential Forward Selection, see
`feature_selection_wrapper_sfs.py` script.
'''
import sys
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
sys.path.append('auxiliary/')  # noqa: E402
from data_processing import load_features, shape_data
from kfold_crosseval import kfold_crosseval


# %% Choose settings and classifier
test_season = 2019  # hold-out season for validation
level = 'match'  # match or team level features to use
shuffle = True  # whether to shuffle or not the data in k-fold cross validation
norm = True  # whether to normalise or not the features
min_round = 5  # minimum number of first rounds to skip in every season
nsplits = 5  # number of folds in k-fold cross validation
nestimators = 188  # this is a classifier-specific setting
rate = 1.2  # this is a classifier-specific setting
random_state = 10  # random state for the classifier
model = AdaBoostClassifier(n_estimators=nestimators, random_state=random_state,
                           learning_rate=rate)

# %% load feature data
df = load_features(level)

# %% choose features
if level == 'match':
    feats = ['Position_x', 'Position_y', 'Offence_x', 'Offence_y',
             'Defence_x', 'Defence_y',
             'form_x', 'form_y',
             'Diff_x', 'Diff_y',
             'Home F4', 'Away F4']
elif level == 'team':
    feats = ['Home', 'Away', 'Position', 'Offence', 'Defence',
             'form', 'F4', 'Diff']
n_feats = len(feats)

# seasons for calibration
df = df[df['Season'] < test_season]

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

    scores[ii, 0], scores[ii, 1] = kfold_crosseval(X_train, y_train, df[comb],
                                                   nsplits, groups=groups,
                                                   model=model,
                                                   level=level,
                                                   shuffle=shuffle)
# save results
np.savez('output/wrapper_ada2_n_{}_rate_{}'.format(nestimators, rate),
         scores=scores, features=np.array(allcombs))

# %% Plot results
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
