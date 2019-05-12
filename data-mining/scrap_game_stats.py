# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 20:08:28 2019

@author: Georgios
"""
import argparse
import os
from bs4 import BeautifulSoup
import requests
import sys
from datetime import datetime
import re
import pandas as pd


def main(season, filename):
    regex = re.compile(r'score [a-z\s]*pts[a-z\s]')
    allteamstats = []
    header = ['Season', 'Round', 'Date', 'Team', 'Where', 'Offence', 'Defence']

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
            home_score = int(game.find_all('span',
                                           attrs={'class': regex})[0].string)
            away_score = int(game.find_all('span',
                                           attrs={'class': regex})[1].string)
            date_str = game.find('span', attrs={'class': 'date'}).string
            date = datetime.strptime(date_str, '%B %d %H:%M CET')
            yr = season if date.month <= 12 and date.month > 8 else season + 1
            date = date.replace(year=yr)
            date_str = datetime.strftime(date, '%Y-%m-%d %H:%M:%S')

            home = {'Season': str(season) + '-' + str(season+1),
                    'Round': game_round,
                    'Date': date_str, 'Team': home_team, 'Where': 'Home',
                    'Offence': home_score, 'Defence': away_score}
            away = {'Season': str(season) + '-' + str(season+1),
                    'Round': game_round,
                    'Date': date_str, 'Team': away_team, 'Where': 'Away',
                    'Offence': away_score, 'Defence': home_score}

            # follow the game-centre link
            link = (game.find_all('a', attrs={'class': 'game-link'})[0]
                    .attrs['href'])
            fulllink = 'http://www.euroleague.net/' + link
            try:
                r = requests.get(fulllink)
            except ConnectionError:
                sys.exit('Connection Error. Check Game URL')
            gamedata = r.text
            gamesoup = BeautifulSoup(gamedata, 'html.parser')
            totals = gamesoup.find_all('tr', attrs={'class': 'TotalFooter'})
            for i, t in enumerate(totals):
                if i == 0:
                    # home team stats
                    dics = home.copy()
                elif i == 1:
                    # away team stats
                    dics = away.copy()
                else:
                    print('Totals field returned invalid number of teams')
                stats = t.find_all('span')
                for stat in stats:
                    # ignore total time played field
                    fullfield = stat.attrs['id']
                    if 'TotalTimePlayed' not in fullfield:
                        ii = fullfield.find('_lbl')
                        field = fullfield[ii+9:]
                        string = stat.contents[0]
                        if string.isnumeric():
                            f = int(string)
                            dics[field] = f
                            if field not in header:
                                header.append(field)
                        elif '/' in string:
                            made, attmp = string.split('/')
                            dics[field+'-Made'] = int(made)
                            dics[field+'-Attempted'] = int(attmp)
                            if field+'-Made' not in header:
                                header.append(field+'-Made')
                            if field+'-Attempted' not in header:
                                header.append(field+'-Attempted')
                        else:
                            print('Invalid field value')
                allteamstats.append(dics)

    print('Convert to dataframe')
    df = pd.DataFrame(allteamstats, columns=header)
    print('Save to file')
    df.to_csv(filename, index=False)
    return df


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
