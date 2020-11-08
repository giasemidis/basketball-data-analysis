'''
Wrapper method for feature selection using the  Sequential Forward Selection
using a chosen algorithm and its hyper-parameters.
'''
import sys
import numpy as np
from matplotlib import pyplot as plt
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
nestimators = 115  # this is a classifier-specific setting
rate = 1.1  # this is a classifier-specific setting
random_state = 10  # random state for the classifier
model = AdaBoostClassifier(n_estimators=nestimators, random_state=random_state,
                           learning_rate=rate)

# %% load feature data
df = load_features(level)

# choose features
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

print([feats[u] for u in bestfeats])

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
