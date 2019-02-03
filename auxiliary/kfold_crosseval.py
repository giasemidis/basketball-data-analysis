# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 22:51:12 2019

@author: Georgios
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from auxiliary.sample_weights import sample_weights
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def kfold_crosseval(X_train, y_train, df_train, nsplits, groups=None, 
          model=GaussianNB(), level='match', shuffle=True):
    if level == 'team':
        kfold = GroupKFold(n_splits=nsplits)
        folditer = kfold.split(X_train, y_train, groups)
    else:
        kfold = StratifiedKFold(n_splits=nsplits, shuffle=shuffle, random_state=10)
        folditer = kfold.split(X_train, y_train)

    accuracy = np.zeros(kfold.get_n_splits())
    w_accuracy = np.zeros(kfold.get_n_splits())     
    i = -1
    for train_index, test_index in folditer:
        #loop over folds
        i+=1
#        print(i)
        X_train_folds, X_test_fold = X_train[train_index, :], X_train[test_index, :]
        y_train_folds, y_test_fold = y_train[train_index], y_train[test_index]
        df_test_fold = df_train.iloc[test_index, :].copy()
        w = sample_weights(y_test_fold, 2)

#        model = GaussianNB()
        # fit model
        model.fit(X_train_folds, y_train_folds)
    
        if level == 'team':
            # calculate accuracy at the match level
            y_pred_prob = model.predict_proba(X_test_fold)
            df_test_fold['Prob'] = y_pred_prob[:, 1]
            y_test_fold = []
            y_pred = []
            for gid in np.unique(df_test_fold['Game ID']):
                teams = df_test_fold[df_test_fold['Game ID']==gid]
                if teams.shape[0] == 2:
                    game_pred = 1 if teams.iloc[0]['Prob'] > teams.iloc[1]['Prob'] else 0
                    game_resu = 1 if teams.iloc[0]['Label'] > teams.iloc[1]['Label'] else 0
                    y_test_fold.append(game_resu)
                    y_pred.append(game_pred)
                else:
                    print('Warning: Game ID %d has missing teams' % gid)
            y_test_fold = np.array(y_test_fold)
            y_pred = np.array(y_pred)
            w = sample_weights(y_test_fold, 2)
        else:
            # predict model
            y_pred = model.predict(X_test_fold)
            
        accuracy[i] = accuracy_score(y_test_fold, y_pred)
        w_accuracy[i] = accuracy_score(y_test_fold, y_pred, sample_weight=w)
    return accuracy.mean(), w_accuracy.mean()
    
