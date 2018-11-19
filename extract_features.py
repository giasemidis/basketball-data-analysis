# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 02:17:41 2018

@author: Georgios
"""
import numpy as np
import pandas as pd
import sys
from auxiliary.make_features import make_features_from_df
from auxiliary.make_features import make_team_features

year = 2017
data = pd.read_csv('data/euroleague_results_%d_%d.csv' % (year, year+1))
standings = pd.read_csv('data/euroleague_standings_%d_%d.csv' % (year, year+1))

#%%
label = np.where(data['Home Score'] > data['Away Score'], 1, 2)

feats = make_features_from_df(data, standings)

feats.insert(3, 'Label', label)
feats.to_csv('data/match_level_features_%d_%d.csv' % (year, year+1),
             index=False)

#%%

team_feats = make_team_features(data, standings)
team_feats.to_csv('data/team_level_features_%d_%d.csv' % (year, year+1), 
                  index=False)