# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:52:26 2018

@author: Georgios
"""
import argparse
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import sys
from datetime import datetime


def main(season):
    '''
    Scraps the results of the Euroleague games from the Euroleague's official
    site for the input season.
    Saves data to file.
    '''

    headers = ['Round', 'Date', 'Home Team', 'Away Team', 
               'Home Score', 'Away Score']
    results  = []
    regex = re.compile('score [a-z\s]*pts[a-z\s]')
    bb = []
    for game_round in range(1, 31):
        print('Processing round %d' % game_round)
        url = 'http://www.euroleague.net/main/results?gamenumber=%d&phasetypecode=RS&seasoncode=E%d' % (game_round, season)
        try:
            r  = requests.get(url)
        except ConnectionError:
            sys.exit('Connection Error. Check URL')
        data = r.text
        soup = BeautifulSoup(data, 'html.parser')
        for game in soup.find_all('div', attrs={'class': 'game played'}):
            home_team = game.find_all('span', attrs={'class': 'name'})[0].string
            away_team = game.find_all('span', attrs={'class': 'name'})[1].string
            home_score = int(game.find_all('span', attrs={'class': regex})[0].string)
            away_score = int(game.find_all('span', attrs={'class': regex})[1].string)
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
    df.to_csv('../data/euroleague_results_%s2.csv' % season, index=False)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--season', type=int,
                        help="the season year")
    args = parser.parse_args()
    
    if args.season is None:
        parser.print_help()
    else:
        main(args.season)