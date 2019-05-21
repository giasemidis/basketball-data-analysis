# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 22:30:59 2018

@author: Georgios
"""
import numpy as np
import pandas as pd
import sys
from itertools import permutations


def make_standings(results, nround):

    if nround < 1:
        sys.exit('Game round must be greater than 0')

    results = results[results['Game Round'] <= nround].copy()
    home_points = np.ones(results.shape[0], dtype=int)
    away_points = np.ones(results.shape[0], dtype=int)
    jj = results['Home Score'] > results['Away Score']
    home_points[jj] = 2
    away_points[np.logical_not(jj)] = 2

    results['Home Points'] = home_points
    results['Away Points'] = away_points

    home = results.groupby(['Home Team ID'])['Home Points',
                                             'Home Score Regular Period',
                                             'Away Score Regular Period'].sum()
    away = results.groupby(['Away Team ID'])['Away Points',
                                             'Away Score Regular Period',
                                             'Home Score Regular Period'].sum()

    groupby = home.merge(away, how='outer', left_index=True, right_index=True)
    groupby.fillna(0, inplace=True)

    teamids = np.concatenate((results['Home Team ID'].values,
                              results['Away Team ID'].values), axis=0)
    teams = np.concatenate((results['Home Team'].values,
                            results['Away Team'].values), axis=0)
    dct = dict(zip(teamids, teams))

    standing = pd.DataFrame()
    standing['Team ID'] = groupby.index
    standing['Team'] = [dct[u] for u in standing['Team ID'].values]
    standing['Points'] = (groupby['Home Points'].values +
                          groupby['Away Points'].values)
    standing['Score+'] = (groupby['Home Score Regular Period_x'].values +
                          groupby['Away Score Regular Period_y'].values)
    standing['Score-'] = (groupby['Away Score Regular Period_x'].values +
                          groupby['Home Score Regular Period_y'].values)
    standing['Score Diff'] = standing['Score+'] - standing['Score-']
    standing.sort_values(by=['Points', 'Score Diff', 'Score+'], inplace=True,
                         ascending=False)
    standing.reset_index(drop=True, inplace=True)

    intcols = ['Team ID', 'Points', 'Score+', 'Score-', 'Score Diff']
    standing[intcols] = standing[intcols].astype(int)

    if nround < standing.shape[0]:
        return standing

    nteams = standing.shape[0]
    secondary_points = np.zeros(nteams, dtype=int)
    score_diffs = np.zeros(nteams, dtype=int)
    for p in np.unique(standing['Points'].values):
        if np.sum(standing['Points'].values == p) > 1:

            # there are ties
            kk = standing['Points'].values == p
            teams = standing['Team ID'].iloc[kk]
            ii = (np.in1d(results['Home Team ID'].values, teams) &
                  np.in1d(results['Away Team ID'].values, teams))
            minichamp = results.iloc[ii]
            home = minichamp.groupby(['Home Team ID'])[
                'Home Points', 'Home Score Regular Period',
                'Away Score Regular Period'].sum()
            away = minichamp.groupby(['Away Team ID'])[
                'Away Points', 'Away Score Regular Period',
                'Home Score Regular Period'].sum()
            groupby = home.merge(away, how='outer', left_index=True,
                                 right_index=True)

            # only those tied teams that have played
            # all against each other twice are ordered
            # by their head-to-head matches.
            flag = True
            for h, a in permutations(groupby.index, 2):
                if any((results['Home Team ID'].values == h) &
                       (results['Away Team ID'].values == a)):
                    pass
                else:
                    flag = False

            if flag is False:
                continue

            groupby.fillna(0, inplace=True)

            teamid = groupby.index
            points = (groupby['Home Points'].values +
                      groupby['Away Points'].values)
            scoreplus = (groupby['Home Score Regular Period_x'].values +
                         groupby['Away Score Regular Period_y'].values)
            scoreminus = (groupby['Away Score Regular Period_x'].values +
                          groupby['Home Score Regular Period_y'].values)
            scores = scoreplus - scoreminus

            for team, point, score in zip(teamid, points, scores):
                secondary_points[standing['Team ID'] == team] = point
                score_diffs[standing['Team ID'] == team] = score

    standing['Secondary Points'] = secondary_points
    standing['Secondary Score Diff'] = score_diffs
    standing.sort_values(by=['Points', 'Secondary Points',
                             'Secondary Score Diff', 'Score Diff'],
                         inplace=True, ascending=False)
    standing.reset_index(drop=True, inplace=True)

    return standing
