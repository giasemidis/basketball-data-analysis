# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:55:18 2018

@author: Georgios
"""
import pandas as pd
import numpy as np

   
def fix_team_names(df1, df2):


#    name_dict = {'AX Armani Exchange Olimpia Milan': 'Ea7-Emporio Armani Milano',
#                 'Anadolu Efes Istanbul': 'Efes Anadolu Istanbul',
#                 'Crvena Zvezda mts Belgrade': 'KK Crvena Zvezda MTS',
#                 'FC Barcelona Lassa': 'FC Barcelona',
#                 'Fenerbahce Dogus Istanbul': 'Fenerbahce Istanbul',
#                 'KIROLBET Baskonia Vitoria Gasteiz': 'Baskonia Vitoria Gasteiz',
#                 'Khimki Moscow Region': 'BC Khimki Moscow',
#                 'Maccabi FOX Tel Aviv': 'Maccabi Tel-Aviv',
#                 'Olympiacos Piraeus': 'BC Olympiakos Piraeus',
#                 'Panathinaikos Superfoods Athens': 'Panathinaikos',
#                 'Real Madrid': 'Real Madrid BC',
#                 'Zalgiris Kaunas': 'BC Zalgiris Kaunas'}

    name_dict = {'EA7 Emporio Armani Milan': 'AX Armani Exchange Olimpia Milan',
                 'Fenerbahce Istanbul': 'Fenerbahce Dogus Istanbul',
                 'Baskonia Vitoria Gasteiz': 'KIROLBET Baskonia Vitoria Gasteiz'}
    
    for team in name_dict.keys():
        print(team in df1['Home Team'].values)
        df1.replace(team, name_dict[team], inplace=True)
    
    teams1 = np.unique(df1['Home Team'])
    teams2 = np.unique(df2['Home Team'])
    if not np.in1d(teams1, teams2).all():
        ii = ~np.in1d(teams1, teams2)
        print(teams1[ii])
    if not np.in1d(teams2, teams1).all():
        ii = ~np.in1d(teams2, teams1)
        print(teams2[ii])

    return df1, df2
