# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 02:17:41 2018

@author: Georgios
"""
import numpy as np
import pandas as pd
import sys
from auxiliary.make_features import make_features_from_df

year = 2016
data = pd.read_csv('data/euroleague_results_%d_%d.csv' % (year, year+1))
standings = pd.read_csv('data/euroleague_standings_%d_%d.csv' % (year, year+1))

feats = make_features_from_df(data, standings)
feats.to_csv('data/match_level_features_%d_%d.csv' % (year, year+1), index=False)
