import os
import sys
import argparse
import logging
import re
from datetime import datetime
from tqdm import trange
from bs4 import BeautifulSoup
import requests
import pandas as pd
sys.path.append('auxiliary/')  # noqa: E402
from io_json import read_json

logging.basicConfig(level=logging.INFO)


def main(season, n_rounds):
    '''
    Scraps the results of the Euroleague games from the Euroleague's official
    site for the input season.
    Saves data to file.
    '''

    # read settings
    settings = read_json('settings/data_collection.json')
    out_dir = settings['output_dir']
    url_pattern = settings['season_results']['url_link']
    out_file_prefix = settings['season_results']['output_file_prefix']
    filename = '%s_%d_%d.csv' % (out_file_prefix, season - 1, season)
    filepath = os.path.join(out_dir, filename)

    headers = ['Season', 'Round', 'GameID', 'Date', 'Home Team', 'Away Team',
               'Home Score', 'Away Score']
    results = []
    regex = re.compile(r'score [a-z\s]*pts[a-z\s]*')
    season_str = '%d-%d' % (season - 1, season)
    for game_round in trange(1, n_rounds + 1):

        url = (url_pattern % (game_round, season - 1))
        try:
            r = requests.get(url)
        except ConnectionError:
            sys.exit('Connection Error. Check URL')
        data = r.text
        soup = BeautifulSoup(data, 'html.parser')
        for game in soup.find_all('div', attrs={'class': 'game played'}):
            data_code = game.attrs['data-code']
            gameid = '%d_%d_%d_%s' % (season - 1, season,
                                      game_round, data_code)
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
            yr = season - 1 if date.month <= 12 and date.month > 8 else season
            date = date.replace(year=yr)
            date_str = datetime.strftime(date, '%Y-%m-%d %H:%M:%S')

            results.append([season_str, game_round, gameid, date_str,
                            home_team, away_team,
                            home_score, away_score])

    logging.info('Convert to dataframe')
    df = pd.DataFrame(results, columns=headers)

    logging.info('Save to file')
    df.to_csv(filepath, index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--season', required=True, type=int,
                        help="the ending year of the season")
    parser.add_argument('-n', '--n-rounds', default=34,
                        type=int,
                        help="The number of regular season rounds "
                             "in the season")
    args = parser.parse_args()

    main(args.season, args.n_rounds)
