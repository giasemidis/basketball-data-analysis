import argparse
from tqdm import trange
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import sys
from datetime import datetime
sys.path.append('auxiliary/')  # noqa: E402
from argparser_types import is_valid_parent_path


def main(season, n_rounds, filename):
    '''
    Scraps the results of the Euroleague games from the Euroleague's official
    site for the input season.
    Saves data to file.
    '''
    headers = ['Season', 'Round', 'GameID', 'Date', 'Home Team', 'Away Team',
               'Home Score', 'Away Score']
    results = []
    regex = re.compile(r'score [a-z\s]*pts[a-z\s]*')
    season_str = '%d-%d' % (season - 1, season)
    for game_round in trange(1, n_rounds + 1):
        # print('Processing round %d' % game_round)
        url = ('http://www.euroleague.net/main/results?gamenumber=%d&'
               'phasetypecode=RS&seasoncode=E%d' % (game_round, season - 1))
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

    print('Convert to dataframe')
    df = pd.DataFrame(results, columns=headers)
    print('Save to file')
    df.to_csv(filename, index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--season', required=True, type=int,
                        help="the ending year of the season")
    parser.add_argument('-o', '--output', required=True,
                        type=lambda x: is_valid_parent_path(parser, x),
                        help="the full filepath of the output file")
    parser.add_argument('-n', '--n-rounds', default=34,
                        type=int,
                        help="The number of regular season rounds "
                             "in the season")
    args = parser.parse_args()

    main(args.season, args.n_rounds, args.output)
