# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:27:06 2018

@author: Georgios

Performs simple analysis and evaluates the scoring of simple benchmark models:
    1) Home team always wins
    2) F4 teams alway win when playing with a non-F4 team,
        otherwise home team always wins.
    3) Persistence model, teams that won in the previoous round win,
        if both teams have won, home team wins.
    4) Standing model, team higher in the standings wins.
    5) Panathinaikos always wins, otherwise home team always wins
    6) Random model.
"""
import argparse
import numpy as np
import pandas as pd
import sys
sys.path.append('auxiliary')
from io_json import read_json


def main(year):

    # read input data (results and standings)
    data = pd.read_csv('data/euroleague_results_%d_%d.csv' % (year, year + 1))
    standings = pd.read_csv('data/euroleague_standings_%d_%d.csv'
                            % (year, year + 1))
    f4teams = read_json('data/f4teams.json')

    # Specify the F4 teams of the previous year
    f4Teams = f4teams[str(year)]

    # Checks
    flag = False
    stand_teams = np.unique(standings['Club Name'])
    resul_teams = np.unique(data['Home Team'])
    if not np.in1d(stand_teams, resul_teams).all():
        ii = ~np.in1d(stand_teams, resul_teams)
        print(stand_teams[ii])
        flag = True
    if not np.in1d(resul_teams, stand_teams).all():
        ii = ~np.in1d(resul_teams, stand_teams)
        print(resul_teams[ii])
        flag = True

    if flag:
        sys.exit('Fix inconsistancies in team names')

    # teams = np.unique((data['Home Team'].values))
    nmatches = data.shape[0]
    # ngames = int(2 * nmatches / len(teams))

    # team_dict = dict.fromkeys(teams, np.ones(ngames, dtype=int))

    data['Actual'] = np.where(data['Home Score'] > data['Away Score'], 1, 2)
    data['Home Wins'] = np.ones(nmatches, dtype=int)

    # f4 model: the F4 teams of the previous year always win.
    # If no or both F4 teams in a game, home always wins.
    f4wins = np.ones(nmatches, dtype=int)
    hmf4 = np.in1d(data['Home Team'], f4Teams)
    awf4 = np.in1d(data['Away Team'], f4Teams)
    f4wins[awf4 & (~hmf4)] = 2
    data['F4 Wins'] = f4wins

    # persistence model: a team that won the previous games wins. If no or both
    # teams won the last game, home always wins.
    # standings model: the team that is higher in the standings wins.
    persistence = np.ones(nmatches, dtype=int)
    stand = np.ones(nmatches, dtype=int)
    for r in np.unique(data['Round']):
        if r == 1:
            continue

        # standings model
        s = standings[standings['Round'] == r - 1]
        d = data[data['Round'] == r]

        home_stands = np.array([s[s['Club Name'] == u]['Position'].iloc[0]
                                for u in d['Home Team']])
        away_stands = np.array([s[s['Club Name'] == u]['Position'].iloc[0]
                                for u in d['Away Team']])
        stand[data['Round'] == r] = np.where(home_stands < away_stands, 1, 2)

        # persistence model
        if r == 2:
            home_won = np.array([1 if s[s['Club Name'] == u]['Wins'].iloc[0] > 0
                                else 0 for u in d['Home Team']])
            away_won = np.array([1 if s[s['Club Name'] == u]['Wins'].iloc[0] > 0
                                else 0 for u in d['Away Team']])
        else:
            s_prev = standings[standings['Round'] == r - 2]
            home_won = np.array([1 if s[s['Club Name'] == u]['Wins'].iloc[0] >
                                 s_prev[s_prev['Club Name'] == u]['Wins']
                                 .iloc[0]
                                 else 0 for u in d['Home Team']])
            away_won = np.array([1 if s[s['Club Name'] == u]['Wins'].iloc[0] >
                                 s_prev[s_prev['Club Name'] == u]['Wins']
                                 .iloc[0]
                                 else 0 for u in d['Away Team']])
        persistence[data['Round'] == r] = np.where(away_won > home_won, 2, 1)

    data['Standings'] = stand
    data['Persistence'] = persistence
    # Pana model: Pana always wins, in any other game, home always wins
    data['Pana'] = np.where(data['Away Team'] ==
                            'Panathinaikos Superfoods Athens', 2, 1)

    # Random model, for 1000 iterations, randomly assign the results of
    # the games
    random = np.zeros(1000)
    for i in range(1000):
        rand = np.random.randint(1, 3, nmatches)
        random[i] = np.sum(data['Actual'].values == rand)

    # data = data[~np.in1d(data['Round'], [1, 2, 3, 4, 5, 26])]
    print('Home wins  :', np.sum(data['Actual'] == data['Home Wins']))
    print('Top4 wins  :', np.sum(data['Actual'] == data['F4 Wins']))
    print('Persistance:', np.sum(data['Actual'] == data['Persistence']))
    print('Standing   :', np.sum(data['Actual'] == data['Standings']))
    print('Pana       :', np.sum(data['Actual'] == data['Pana']))
    print('Random     :', np.round(np.mean(random), 0))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--season', type=int,
                        help="the begingin year of a season")
    args = parser.parse_args()

    if args.season is None:
        parser.print_help()
    else:
        main(args.season)
