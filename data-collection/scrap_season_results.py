# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:52:26 2018

@author: Georgios
"""
import argparse
import os
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import sys
from datetime import datetime


def main(season, filename):
    '''
    Scraps the results of the Euroleague games from the Euroleague's official
    site for the input season.
    Saves data to file.
    '''
    season = season - 1
    headers = ['Round', 'Date', 'Home Team', 'Away Team',
               'Home Score', 'Away Score']
    results = []
    regex = re.compile(r'score [a-z\s]*pts[a-z\s]*')
    bb = []
    for game_round in range(1, 31):
        print('Processing round %d' % game_round)
        url = ('http://www.euroleague.net/main/results?gamenumber=%d&'
               'phasetypecode=RS&seasoncode=E%d' % (game_round, season))
        try:
            r = requests.get(url)
        except ConnectionError:
            sys.exit('Connection Error. Check URL')
        data = r.text
        soup = BeautifulSoup(data, 'html.parser')
        for game in soup.find_all('div', attrs={'class': 'game played'}):
            home_team = game.find_all('span', attrs={'class': 'name'})[0].string
            away_team = game.find_all('span', attrs={'class': 'name'})[1].string
            scores = game.find_all('span', attrs={'class': regex})

            home_score = int(scores[0]['data-score'] if
                             scores[0].has_attr('data-score') else
                             scores[0].string)
            away_score = int(scores[1]['data-score'] if
                             scores[1].has_attr('data-score') else
                             scores[1].string)

            date_str = game.find('span', attrs={'class': 'date'}).string
            date = datetime.strptime(date_str, '%B %d %H:%M CET')
            if date.month <= 12 and date.month > 8:
                yr = season
            else:
                yr = season + 1
            date = date.replace(year=yr)
            date_str = datetime.strftime(date, '%Y-%m-%d %H:%M:%S')
            status = game.find('span', attrs={'class': 'final'}).string.strip()

            bb.append(status)
            results.append([game_round, date_str, home_team, away_team,
                            home_score, away_score])

    print('Convert to dataframe')
    df = pd.DataFrame(results, columns=headers)
    print('Save to file')
    df.to_csv(filename, index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--season', type=int,
                        help="the ending year of the season")
    parser.add_argument('-o', '--output', type=str,
                        help="the full filepath of the output file")
    args = parser.parse_args()

    if args.season is None or args.output is None:
        parser.print_help()
    else:
        if not os.path.isdir(os.path.split(args.output)[0]):
            sys.exit('Warning: path of output file not valid.')
        else:
            main(args.season, args.output)
