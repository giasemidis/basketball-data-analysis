# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 02:17:41 2018

@author: Georgios
"""
import numpy as np
import pandas as pd
import sys
from auxiliary.make_standings import make_standings
from auxiliary.make_features import  make_features

df = pd.read_csv('data/euroleague_season_2018_results-end.csv')
#for i in range(1, 31):
#    print(i)
#    standing = make_standings(df, i)

standings = {}
for i in range(1, 31):
    standings[i] = make_standings(df, i)


features_df = make_features(df, standings=standings)
features = features_df.values

target = np.ones(features.shape[0], dtype=int)
target[df['Home Score'] < df['Away Score']] = 2

features_df.insert(loc=0, column='mach_id', value=df['Match ID'])
features_df.insert(loc=1, column='label', value=target)

features_df.to_csv('data/match_level_features.csv', sep=',', index=False)

x = np.concatenate((target[:, np.newaxis], features), axis=1)
np.savetxt('data/match_level_features_raw.csv', x, delimiter=',')
