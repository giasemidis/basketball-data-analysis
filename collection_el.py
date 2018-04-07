# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 19:15:10 2018

@author: Georgios
"""
from auxiliary.io_csv import write_csv
from auxiliary.sports_radar_api import sports_radar_api
from auxiliary.results_strcted import results_strcted

methods = {'tournaments': 
    {'id': '',
     'submethod': 'info',
     'options': ['info', 'live_standings', 'results', 'schedule', 'standings']}}

#r = sports_radar_api('q6jmutjvbxwk7eg69ypu73fx', 'tournaments.json')
#print(r.keys())
#for i in r['tournaments']:
#    if i['name'] == 'Euroleague':
#        print(i['name'], i['id'], i['sport']['name'], i['current_season']['name'])

r = sports_radar_api('q6jmutjvbxwk7eg69ypu73fx',
                     'tournaments/sr:tournament:138/results.json')
#r = sports_radar_api('q6jmutjvbxwk7eg69ypu73fx',
#                     'tournaments/sr:tournament:138/standings.json')

data, header = results_strcted(r)

write_csv('output/euroleague_season_2018_results-end.csv', data, header)