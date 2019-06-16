# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:34:10 2019

@author: Georgios
"""
import numpy as np
import sys
import pandas as pd
from normalise import normalise


def shape_data(df, feats, norm=True, min_round=5):

    # ignore early games in the season, as they do not contain the 'form'
    # feature.
    ii = df['Round'] > min_round

    # filter out the games ignored
    df = df[ii]
    df.reset_index(drop=True, inplace=True)

    # make the Design table
    X_train = df[feats].values

    # normalise the Design table if required
    if norm:
        X_train = normalise(X_train)

    # extract the tags
    y_train = df['Label'].values

    # if labels are 1 and 2, set them to 0-1
    if 2 in np.unique(y_train):
        y_train = y_train - 1

    # define the groups matches if processing 'team' level classification
    groups = df['Game ID'].values if 'Game ID' in df.keys() else None

    return X_train, y_train, df, groups


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
    seasoncol = np.concatenate(((2017 * np.ones(df1.shape[0], dtype=int)),
                                (2018 * np.ones(df2.shape[0], dtype=int)),
                                (2019 * np.ones(df3.shape[0], dtype=int))),
                               axis=0)
    if 'Season' not in df.keys():
        df.insert(1, 'Season', seasoncol)
    df.reset_index(drop=True, inplace=True)
    return df
