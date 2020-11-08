'''
Hyper-parameter tuning using k-fold cross-validation for one hyper-parameter
models via loops grid search. This scripts is left for legacy, see
also the `gridsearch_validation.py` which covers multiple hyper-parameter
models.
'''
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
sys.path.append('auxiliary/')  # noqa: E402
from data_processing import load_features, shape_data
from kfold_crosseval import kfold_crosseval


# %% Choose settings and classifier
test_season = 2019  # hold-out season for validation
level = 'team'  # match or team level features to use
shuffle = True  # whether to shuffle or not the data in k-fold cross validation
norm = True  # whether to normalise or not the features
min_round = 5  # minimum number of first rounds to skip in every season
nsplits = 5  # number of folds in k-fold cross validation
method = 'log-reg'  # method for grid search hyper-parameter training, see list
# methods: 'log-reg', 'svm-linear', 'svm-rbf', 'decision-tree', 'random-forest',
# 'naive-bayes', 'gradient-boosting', 'ada', 'ada2', 'knn',
# 'discriminant-analysis'

print('level: %s - norm: %r - shuffle: %r - method: %s' %
      (level, norm, shuffle, method))

# %% load feature data
df = load_features(level)

# choose features
if level == 'match':
    feats = ['Position_x', 'Position_y', 'Offence_x', 'Offence_y',
             'Defence_x', 'Defence_y', 'form_x', 'form_y', 'Diff_x', 'Diff_y',
             'Home F4', 'Away F4']
elif level == 'team':
    feats = ['Home', 'Away', 'Position',
             'Offence', 'Defence', 'form', 'F4', 'Diff']

# seasons for calibration
df = df[df['Season'] < test_season]

# %% Re-shape data
X_train, y_train, df, groups = shape_data(df, feats, norm=norm,
                                          min_round=min_round)
print('Number of feaures:', X_train.shape[1], feats)
print('Number of obs:', X_train.shape[0])

# %% Set parameters
if method == 'log-reg':
    params = np.sort(np.concatenate((np.logspace(-5, 8, 14),
                                     5 * np.logspace(-5, 8, 14)), axis=0))
elif method == 'svm-linear':
    params = np.sort(np.concatenate((np.logspace(-5, 6, 12),
                                     5 * np.logspace(-5, 6, 12)), axis=0))
elif method == 'decision-tree':
    params = np.array([0])
elif method == 'random-forest':
    params = np.arange(10, 100, 5)
elif method == 'naive-bayes':
    params = np.array([0])
elif method == 'gradient-boosting':
    params = np.arange(10, 200, 10)
elif method == 'ada':
    params = np.arange(5, 200, 3)
elif method == 'knn':
    params = np.arange(3, 30, 2)
elif method == 'discriminant-analysis':
    params = np.array([0])
else:
    sys.exit('Method not recognised')

# %% Tune parameters
accuracy = np.zeros(params.shape[0])
w_accuracy = np.zeros(params.shape[0])

for j, param in enumerate(params):

    # update model's parameters
    if method == 'log-reg':
        model = LogisticRegression(C=param, solver='liblinear',
                                   class_weight='balanced')
    elif method == 'svm-linear':
        model = SVC(C=param, kernel='linear', class_weight='balanced',
                    probability=True)
    elif method == 'decision-tree':
        model = DecisionTreeClassifier(class_weight='balanced', random_state=10)
    elif method == 'random-forest':
        model = RandomForestClassifier(n_estimators=param,
                                       class_weight='balanced',
                                       random_state=10)
    elif method == 'naive-bayes':
        model = GaussianNB()
    elif method == 'gradient-boosting':
        model = GradientBoostingClassifier(n_estimators=param, random_state=10)
    elif method == 'ada':
        model = AdaBoostClassifier(n_estimators=param, random_state=10,
                                   learning_rate=0.6)
    elif method == 'knn':
        model = KNeighborsClassifier(n_neighbors=param)
    elif method == 'discriminant-analysis':
        model = QuadraticDiscriminantAnalysis()
    else:
        sys.exit('method name is not valid')

    # apply k-fold cross validation
    accuracy[j], w_accuracy[j] = kfold_crosseval(X_train, y_train,
                                                 df, nsplits, groups=groups,
                                                 model=model, level=level,
                                                 shuffle=shuffle)

# %% Plots
if params.shape[0] > 1:
    print('Accuracy: ', np.round(np.max(accuracy), 4))
    print('Weighted Accuracy: ', np.round(np.max(w_accuracy), 4))
    plt.figure()
    plt.plot(params, accuracy, label='accuracy')
    plt.plot(params, w_accuracy, label='w_accuracy')
    if method in ['log-reg', 'svm-linear']:
        plt.xscale('log')
    plt.xlabel('parameter')
    plt.ylabel('Score')
    plt.legend()
    plt.title(method)
    plt.show()
else:
    print('Accuracy: ', accuracy.mean(axis=0))
    print('Weighted Accuracy: ', w_accuracy.mean(axis=0))
