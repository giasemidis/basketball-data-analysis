# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:27:06 2018

@author: Georgios
"""
import numpy as np
#from auxiliary.io_csv import read_csv
import pandas as pd

#data, header = read_csv('data/euroleague_results_2017.csv')
data = pd.read_csv('data/euroleague_results_2017.csv')
standings = pd.read_csv('data/euroleague_standings_2017.csv')

#f4Teams = ['Fenerbahce Istanbul', 'BC Olympiakos Piraeus', 'CSKA Moscow', 'Real Madrid BC']
f4Teams = ['Fenerbahce Dogus Istanbul', 'Olympiacos Piraeus', 
           'CSKA Moscow', 'Real Madrid']

teams = np.unique((data['Home Team'].values))
nmatches = data.shape[0]
ngames = int(2 * nmatches / len(teams))

team_dict = dict.fromkeys(teams, np.ones(ngames, dtype=int))

data['Actual'] = np.where(data['Home Score'].values > data['Away Score'].values, 1, 2)
data['Home Wins'] = np.ones(nmatches, dtype=int)

f4wins = np.ones(nmatches, dtype=int)
hmf4 = np.in1d(data['Home Team'].values, f4Teams)
awf4 = np.in1d(data['Away Team'].values, f4Teams)
f4wins[awf4 & (~hmf4)] = 2
data['F4 Wins'] = f4wins

#persistence =  np.ones(nmatches)
#
#stand = np.ones(nmatches, dtype=int)
#for r in np.unique(data['Round']):
#    if r == 1:
#        continue
#    s = standings[standings['Round'] == r]
#    stand[(data['Round'] == r) & (s['Club Name']==data[]]
#    
#
data['Pana'] = np.where(data['Away Team']=='Panathinaikos Superfoods Athens', 2, 1)
#
##team_dict = dict.fromkeys(teams, np.zeros(ngames))
#team_dict = {k: {'wl': np.zeros(ngames, dtype=int),
#                 'scored': np.zeros(ngames, dtype=int),
#                 'conceded': np.zeros(ngames, dtype=int)}
#             for k in teams}
#
#for i, match in enumerate(data):
#
#    gd = int(match[3])
#    ii = gd - 1
#    team_dict[match[1]]['wl'][ii] = 1 if int(match[5]) > int(match[6]) else 0
#    team_dict[match[2]]['wl'][ii] = 1 if int(match[6]) > int(match[5]) else 0
#
#    team_dict[match[1]]['scored'][ii] = match[5]
#    team_dict[match[1]]['conceded'][ii] = match[6]
#    team_dict[match[2]]['scored'][ii] = match[6]
#    team_dict[match[2]]['conceded'][ii] = match[5]
#
#    # persistence model
#    if gd > 1 and team_dict[match[1]]['wl'][ii-1] > team_dict[match[2]]['wl'][ii-1]:
#        persistence[i] = 1
#    elif gd > 1 and team_dict[match[2]]['wl'][ii-1] > team_dict[match[1]]['wl'][ii-1]:
#        persistence[i] = 2
#    else:
#        persistence[i] = 1
#
#    # standing model (the higher ranking team wins, if tied home wins)
#    if gd > 1  and np.sum(team_dict[match[2]]['wl'][:ii]) > np.sum(team_dict[match[1]]['wl'][:ii]):
#        standing[i] = 2
#        standing2[i] = 2
#    elif gd > 1 and np.sum(team_dict[match[2]]['wl'][:ii]) == np.sum(team_dict[match[1]]['wl'][:ii]):
#        home_diff = np.sum(team_dict[match[1]]['scored'][:ii]) -\
#                          np.sum(team_dict[match[1]]['conceded'][:ii])
#        away_diff = np.sum(team_dict[match[2]]['scored'][:ii]) -\
#                          np.sum(team_dict[match[2]]['conceded'][:ii])
#        standing2[i] = 2 if away_diff > home_diff else 1

random = np.zeros(1000)
for i in range(1000):
    rand = np.random.randint(1, 3, nmatches)
    random[i] = np.sum(data['Actual'].values==rand)

print('Home wins  :', np.sum(data['Actual'].values==data['Home Wins']))
print('Top4 wins  :', np.sum(data['Actual'].values==data['F4 Wins']))
#print('Persistance:', np.sum(data['Actual'].values==persistence))
#print('Standing   :', np.sum(data['Actual'].values==standing))
#print('Standing2  :', np.sum(data['Actual'].values==standing2))
print('Pana       :', np.sum(data['Actual']==data['Pana']))
print('Random     :', np.round(np.mean(random), 0))