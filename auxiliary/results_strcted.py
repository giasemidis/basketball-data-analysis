# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 23:17:33 2018

@author: Georgios
"""


def results_strcted(dictionary):
    header = ['Match ID', 'Home Team', 'Away Team', 'Game Round', 'Date',
              'Home Score', 'Away Score', 'Status']
    data = []

    if 'results' not in dictionary.keys():
        print('Error: results not found')
        return None, None

    for match in dictionary['results']:
        row = []
        match_id = match['sport_event']['id']
        for team in match['sport_event']['competitors']:
            if team['qualifier'] == 'home':
                home_team = team['name']
            elif team['qualifier'] == 'away':
                away_team = team['name']
            else:
                print('Error team')
        game_day = match['sport_event']['tournament_round']['number']
        date = match['sport_event']['scheduled']
        home_score = match['sport_event_status']['home_score']
        away_score = match['sport_event_status']['away_score']
        status = match['sport_event_status']['match_status']
        row = [match_id, home_team, away_team, game_day, date,
               home_score, away_score, status]
        data.append(row)
    return data, header