import os
import sys
from glob import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
sys.path.append('auxiliary/')  # noqa: E402
from io_json import read_json


def normalise(X):
    '''
    Normalise the features of the input design matrix `X` across the x=0 axis.
    '''
    x_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return x_norm


def shape_data_scaler(df, feats, norm=True, min_round=5):
    '''
    Shape input data in `df` by selecting the `feats`, excluding rounds and
    normalising if `norm=True`.

    Returns four variables:
    * X_train
    * y_train
    * df (the new df)
    * groups (for defining groups of matches)
    * scaler (the scaler object from normalistion)
    '''
    # ignore early games in the season, as they do not contain the 'form'
    # feature.
    ii = df['Round'] > min_round

    # filter out the games ignored
    df = df[ii]
    df.reset_index(drop=True, inplace=True)

    # make the Design table
    X_train = df[feats].values

    # normalise the Design table if required
    if isinstance(norm, bool) and norm:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        # X_train = normalise(X_train)
    elif isinstance(norm, MinMaxScaler):
        X_train = norm.transform(X_train)
        scaler = norm

    # extract the tags
    y_train = df['Label'].values

    # if labels are 1 and 2, set them to 0-1
    if 2 in np.unique(y_train):
        y_train = y_train - 1

    # define the groups matches if processing 'team' level classification
    groups = df['Game ID'].values if 'Game ID' in df.keys() else None

    return X_train, y_train, df, groups, scaler


def shape_data(df, feats, norm=True, min_round=5):
    '''
    Shape input data in `df` by selecting the `feats`, excluding rounds and
    normalising if `norm=True`.

    Returns four variables:
    * X_train
    * y_train
    * df (the new df)
    * groups (for defining groups of matches)
    '''

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


def load_features(level):
    '''load features'''

    settings = read_json('settings/feature_extraction.json')
    feature_dir = settings['feature_dir']

    if level == 'match':
        file_pattern = settings['match_level_feature_file_prefix']
    elif level == 'team':
        file_pattern = settings['team_level_feature_file_prefix']
    else:
        raise ValueError('Invalid level of analysis: %s' % level)

    filepath = os.path.join(feature_dir, file_pattern)
    feature_files = glob('%s*.csv' % filepath)
    list_dfs = [pd.read_csv(file) for file in feature_files]
    df = pd.concat(list_dfs, ignore_index=False)

    df.reset_index(drop=True, inplace=True)
    return df
