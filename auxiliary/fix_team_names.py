import numpy as np


def fix_team_names(df1, df2):
    '''
    Fix inconsistancies in team names across the seasons
    '''
    name_dict = {
        'EA7 Emporio Armani Milan': 'AX Armani Exchange Olimpia Milan',
        'Fenerbahce Istanbul': 'Fenerbahce Dogus Istanbul',
        'Baskonia Vitoria Gasteiz': 'KIROLBET Baskonia Vitoria Gasteiz'
    }

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
