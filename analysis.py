# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:27:06 2018

@author: Georgios
"""
import numpy as np
from auxiliary.io_csv import read_csv

data, header = read_csv('output/euroleague_season_2018_results.csv')
nmatches = len(data)

topTeams = ['Fenerbahce Istanbul', 'BC Olympiakos Piraeus', 'CSKA Moscow', 'Real Madrid BC']
teams = set([row[1] for row in data])
ngames = int(2 * nmatches / len(teams))

team_dict = dict.fromkeys(teams, np.ones(ngames, dtype=int))

actual = np.ones(nmatches)
homewins = np.ones(nmatches)
top4wins = np.ones(nmatches)
persistence =  np.ones(nmatches)
standing = np.ones(nmatches)
standing2 = np.ones(nmatches)
pana = np.ones(nmatches)

#team_dict = dict.fromkeys(teams, np.zeros(ngames))
team_dict = {k: {'wl': np.zeros(ngames, dtype=int),
                 'scored': np.zeros(ngames, dtype=int),
                 'conceded': np.zeros(ngames, dtype=int)}
             for k in teams}

for i, match in enumerate(data):

    gd = int(match[3])
    ii = gd - 1
    team_dict[match[1]]['wl'][ii] = 1 if int(match[5]) > int(match[6]) else 0
    team_dict[match[2]]['wl'][ii] = 1 if int(match[6]) > int(match[5]) else 0

    team_dict[match[1]]['scored'][ii] = match[5]
    team_dict[match[1]]['conceded'][ii] = match[6]
    team_dict[match[2]]['scored'][ii] = match[6]
    team_dict[match[2]]['conceded'][ii] = match[5]

    actual[i] = 1 if int(match[5]) > int(match[6]) else 2

    # top teams model (top 4 teams wins unless they both against each other)
    if match[1] in topTeams and match[2] not in topTeams:
        top4wins[i] = 1
    elif match[2] in topTeams and match[1] not in topTeams:
        top4wins[i] = 2
    else:
        top4wins[i] = 1

    # persistence model
    if gd > 1 and team_dict[match[1]]['wl'][ii-1] > team_dict[match[2]]['wl'][ii-1]:
        persistence[i] = 1
    elif gd > 1 and team_dict[match[2]]['wl'][ii-1] > team_dict[match[1]]['wl'][ii-1]:
        persistence[i] = 2
    else:
        persistence[i] = 1

    # pana model (Panathinaikos always wins)
    if match[2] == 'Panathinaikos':
        pana[i] = 2

    # standing model (the higher ranking team wins, if tied home wins)
    if gd > 1  and np.sum(team_dict[match[2]]['wl'][:ii]) > np.sum(team_dict[match[1]]['wl'][:ii]):
        standing[i] = 2
        standing2[i] = 2
    elif gd > 1 and np.sum(team_dict[match[2]]['wl'][:ii]) == np.sum(team_dict[match[1]]['wl'][:ii]):
        home_diff = np.sum(team_dict[match[1]]['scored'][:ii]) -\
                          np.sum(team_dict[match[1]]['conceded'][:ii])
        away_diff = np.sum(team_dict[match[2]]['scored'][:ii]) -\
                          np.sum(team_dict[match[2]]['conceded'][:ii])
        standing2[i] = 2 if away_diff > home_diff else 1

random = np.zeros(1000)
for i in range(1000):
    rand = np.random.randint(1, 3, nmatches)
    random[i] = np.sum(actual==rand)

print('Home wins  :', np.sum(actual==homewins))
print('Top4 wins  :', np.sum(actual==top4wins))
print('Persistance:', np.sum(actual==persistence))
print('Standing   :', np.sum(actual==standing))
print('Standing2  :', np.sum(actual==standing2))
print('Pana       :', np.sum(actual==pana))
print('Random     :', np.round(np.mean(random), 0))