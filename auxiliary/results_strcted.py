# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 23:17:33 2018

@author: Georgios
"""


def results_strcted(dictionary):
    header = ['Match ID', 'Home Team', 'Away Team', 'Game Round', 'Date',
              'Home Score', 'Away Score', 'Status', 
              'Home Team ID', 'Away Team ID',
              'Home Score Regular Period', 'Away Score Regular Period']
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
                home_id = team['id'].split(':')[-1]
            elif team['qualifier'] == 'away':
                away_team = team['name']
                away_id = team['id'].split(':')[-1]
            else:
                print('Error team')
        game_day = match['sport_event']['tournament_round']['number']
        date = match['sport_event']['scheduled']
        home_score = match['sport_event_status']['home_score']
        away_score = match['sport_event_status']['away_score']
        status = match['sport_event_status']['match_status']
        home_period_scores = []
        away_period_scores = []
        for period in match['sport_event_status']['period_scores']:
            if period['type'] == 'regular_period':
                home_period_scores.append(period['home_score'])
                away_period_scores.append(period['away_score'])
        home_score_regular_time = sum(home_period_scores)
        away_score_regular_time = sum(away_period_scores)
        row = [match_id, home_team, away_team, game_day, date,
               home_score, away_score, status, home_id, away_id,
               home_score_regular_time, away_score_regular_time]
        data.append(row)
    return data, header