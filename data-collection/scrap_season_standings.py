# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:52:26 2018

@author: Georgios
"""
import argparse
from bs4 import BeautifulSoup
import requests
import re
import os
from tqdm import trange
import pandas as pd
import sys
sys.path.append('auxiliary/')  # noqa: E402
from io_json import read_json


def main(season, n_rounds):
    '''
    Scraps the standings of the Euroleague games from the Euroleague's official
    site for the input season.
    Saves data to file.
    '''

    # read settings
    settings = read_json('settings/data_collection.json')
    out_dir = settings['output_dir']
    url_pattern = settings['season_standings']['url_link']
    out_file_prefix = settings['season_standings']['output_file_prefix']
    filename = '%s_%d_%d.csv' % (out_file_prefix, season - 1, season)
    filepath = os.path.join(out_dir, filename)

    headers = ['Round', 'Position', 'Club Code', 'Club Name', 'Wins', 'Losses',
               'Offence', 'Defence', 'Points Diff']
    standings = []
    for game_round in trange(1, n_rounds + 1):
        # print('Processing round %d' % game_round)
        url = (url_pattern % (game_round, season - 1))
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
            team = re.findall(r'[a-zA-Z\s-]+', pos_team)[0].strip()
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
    df.to_csv(filepath, index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--season', required=True, type=int,
                        help="the ending year of the season")
    parser.add_argument('-n', '--n-rounds', default=34, type=int,
                        help="The number of regular season rounds "
                             "in the season")
    args = parser.parse_args()

    main(args.season, args.n_rounds)
