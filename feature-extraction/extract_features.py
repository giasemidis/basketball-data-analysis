# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 02:17:41 2018

@author: Georgios
"""
import numpy as np
import pandas as pd
import argparse
import sys
from make_features import make_game_features
from make_features import make_team_features
sys.path.append('auxiliary')
from io_json import read_json


def main(year, results_file, standings_file, f4_file, output=''):
    '''
    Extract features (game and team) from the fetched data from the
    Euroleague's site
    '''

    data = pd.read_csv(results_file)
    standings = pd.read_csv(standings_file)
    f4teams = read_json(f4_file)

    # Specify the F4 teams of the previous year
    f4Teams = f4teams[str(year)]

    label = np.where(data['Home Score'] > data['Away Score'], 1, 2)

    # make game features
    feats = make_game_features(data, standings, f4Teams)

    feats.insert(3, 'Label', label)
    # save features to file.
    feats.to_csv(output + 'match_level_features.csv',
                 index=False)

    # make team features
    team_feats = make_team_features(data, standings, f4Teams, year)
    # save features to file
    team_feats.to_csv(output + 'team_level_features.csv',
                      index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--season', type=int,
                        help="starting year of the season")
    parser.add_argument('-r', '--results', type=str,
                        help="results file of a season")
    parser.add_argument('-t', '--table', type=str,
                        help="table/standings file of a season")
    parser.add_argument('-f', '--final_four', type=str,
                        help="final four file of teams of previous a season")
    parser.add_argument('-o', '--output', type=str,
                        help="prefix of output files")
    args = parser.parse_args()

    if args.season is None or args.results is None or args.table is None\
            or args.final_four is None:
        parser.print_help()
    else:
        output = '' if args.output is None else args.output
        main(args.season, args.results, args.table, args.final_four,
             output)
