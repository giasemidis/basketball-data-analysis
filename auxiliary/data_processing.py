# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:34:10 2019

@author: Georgios
"""
import numpy as np
import sys
import pandas as pd
from auxiliary.normalise import normalise


def shape_data(df, norm=True, min_round=5):

    n = list(df.columns).index('Label')
    # features start after the 'Label' column
    init_feat = list(df.keys()[n + 1:])
    n_feats = len(init_feat)
    # ignore early games in the season, as they do not contain the 'form'
    # feature.
    ii = df['Round'].values > min_round
    # make the Design table
    X_train = df.iloc[ii, (n + 1):].values
    # normalise the Design table if required
    if norm:
        X_train = normalise(X_train)
    # extract the tags
    y_train = df[ii]['Label'].values
    # if labels are 1 and 2, set them to 0-1
    if 2 in np.unique(y_train):
        y_train = y_train - 1
    # filter out the games ignored
    df = df[ii]
    df.reset_index(drop=True, inplace=True)

    # define the groups matches if processing 'team' level classification
    groups = df['Game ID'].values if 'Game ID' in df.keys() else None

    return X_train, y_train, df, init_feat, n_feats, groups


def load_data(level):
    '''load features'''
    if level == 'match':
        df1 = pd.read_csv('data/match_level_features_2016_2017.csv')
        df2 = pd.read_csv('data/match_level_features_2017_2018.csv')
        df3 = pd.read_csv('data/match_level_features_2018_2019.csv')
    elif level == 'team':
        df1 = pd.read_csv('data/team_level_features_2016_2017.csv')
        df2 = pd.read_csv('data/team_level_features_2017_2018.csv')
        df3 = pd.read_csv('data/team_level_features_2018_2019.csv')
    else:
        sys.exit('Invalid level of analysis')

    df = pd.concat([df1, df2, df3], ignore_index=False)
    return df
