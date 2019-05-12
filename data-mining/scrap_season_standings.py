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


def main(season, filename):
    '''
    Scraps the standings of the Euroleague games from the Euroleague's official
    site for the input season.
    Saves data to file.
    '''
    headers = ['Round', 'Position', 'Club Code', 'Club Name', 'Wins', 'Losses',
               'Offence', 'Defence', 'Points Diff']
    standings = []
    for game_round in range(1, 31):
        print('Processing round %d' % game_round)
        url = ('http://www.euroleague.net/main/standings?gamenumber=%d&'
               'phasetypecode=RS++++++++&seasoncode=E%d'
               % (game_round, season-1))
        try:
            r = requests.get(url)
        except ConnectionError:
            sys.exit('Connection Error. Check URL')
        data = r.text
        soup = BeautifulSoup(data, 'html.parser')
        tbl_cls = ('table responsive fixed-cols-1 table-left-cols-1 '
                   'table-expand table-striped table-hover table-noborder '
                   'table-centered table-condensed')
        table = soup.find('table', attrs={'class': tbl_cls})
        body = table.find('tbody')
        var1 = 'clubcode='
        var2 = '&seasoncode=E'
        for row in body.find_all('tr'):
            a = row.find('a').get('href')
            cc = a[a.find(var1) + len(var1): a.find(var2)]
            # sc = a[a.find(var2) + len(var2):]
            pos_team = row.find('a').string.strip()
            pos = int(re.findall(r'\d{1,2}', pos_team)[0])
            team = re.findall(r'[a-zA-Z\s]+', pos_team)[0].strip()
            stats = row.find_all('td')
            wins = int(stats[1].string.strip())
            losses = int(stats[2].string.strip())
            points_plus = int(stats[3].string.strip())
            points_minus = int(stats[4].string.strip())
            points_diff = int(stats[5].string.strip())
            standings.append([game_round, pos, cc, team, wins, losses,
                              points_plus, points_minus, points_diff])

    print('Convert to dataframe')
    df = pd.DataFrame(standings, columns=headers)
    print('Save ot file')
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
