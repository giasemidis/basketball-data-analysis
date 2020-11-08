'''
Feature transformation methods using Principal Component Analysis (PCA).

For increasing number of principal components, results are being evaluated for
a chosen algorithm using k-fold cross-validation on the training test.
'''
import sys
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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
sys.path.append('auxiliary/')  # noqa: E402
from data_processing import load_features, shape_data


# %% Choose settings and classifier
test_season = '2018-2019'  # hold-out season for validation
level = 'match'  # match or team level features to use
shuffle = True  # whether to shuffle or not the data in k-fold cross validation
norm = True  # whether to normalise or not the features
min_round = 5  # minimum number of first rounds to skip in every season
nsplits = 5  # number of folds in k-fold cross validation
random_state = 10  # random state for the classifier
params = {
    'n_estimators': np.arange(5, 200, 5),
    # 'learning_rate': np.arange(0.3, 1.5, 0.1)}
}  # params for the grid search
model = AdaBoostClassifier(random_state=random_state)

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
             'form',
             'F4', 'Diff']
n_feats = len(feats)

# seasons for calibration
df = df[df['Season'] != test_season]

# %% Re-shape data
X_train, y_train, df, groups = shape_data(df, feats, norm=norm,
                                          min_round=min_round)

# %% Apply PCA and then k-fold cross validation
XX = X_train.copy()
scores = np.zeros((n_feats, 2))
for n in tqdm(range(n_feats)):
    pca = PCA(n_components=n + 1)
    X_train = pca.fit_transform(XX)
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
plt.xlabel('Number of components')
plt.ylabel('Score')
plt.xticks(x, x)
plt.grid()
plt.legend()
plt.show()
