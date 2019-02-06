# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 02:17:41 2018

@author: Georgios
"""
import numpy as np
import pandas as pd
import argparse
from auxiliary.make_features import make_features_from_df
from auxiliary.make_features import make_team_features
from auxiliary.io_json import read_json

def main(year):
    '''
    Extract features (game and team) from the fetched data from the 
    Euroleague's site
    '''

    data = pd.read_csv('data/euroleague_results_%d_%d.csv' % (year, year+1))
    standings = pd.read_csv('data/euroleague_standings_%d_%d.csv' % (year, year+1))
    f4teams = read_json('data/f4teams.json')
    
    # Specify the F4 teams of the previous year
    f4Teams = f4teams[str(year)]

    label = np.where(data['Home Score'] > data['Away Score'], 1, 2)

    # make game features
    feats = make_features_from_df(data, standings, f4Teams)
    
    feats.insert(3, 'Label', label)
    # save features to file.
    feats.to_csv('data/match_level_features_%d_%d-2.csv' % (year, year+1),
                 index=False)
    
    # make team features  
    team_feats = make_team_features(data, standings, f4Teams, year)
    #save features to file
    team_feats.to_csv('data/team_level_features_%d_%d-2.csv' % (year, year+1), 
                      index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--season', type=int,
                        help="the season year")
    args = parser.parse_args()
    
    if args.season is None:
        parser.print_help()
    else:
        main(args.season)