import os
import sys
import argparse
import logging
import pandas as pd
from make_features import make_game_features
from make_features import make_team_features
sys.path.append('auxiliary')  # noqa: E402
from io_json import read_json

logging.basicConfig(level=logging.INFO)


def main(season):
    '''
    Extract features (game and team) from the fetched data from the
    Euroleague's site
    '''
    # get data settings
    data_settings = read_json('settings/data_collection.json')
    out_dir = data_settings['output_dir']
    rslts_file_prefix = data_settings['season_results']['output_file_prefix']
    results_file = os.path.join(
        out_dir, '%s_%d_%d.csv' % (rslts_file_prefix, season - 1, season))
    stnds_file_prefix = data_settings['season_standings']['output_file_prefix']
    standings_file = os.path.join(
        out_dir, '%s_%d_%d.csv' % (stnds_file_prefix, season - 1, season))
    f4_file = data_settings['f4teams_file']

    # get feature settings
    feat_settings = read_json('settings/feature_extraction.json')
    feature_dir = feat_settings['feature_dir']
    match_level_file_ = feat_settings['match_level_feature_file_prefix']
    team_level_file_ = feat_settings['team_level_feature_file_prefix']
    match_level_file = os.path.join(
        feature_dir, '%s_%d_%d.csv' % (match_level_file_, season - 1, season))
    team_level_file = os.path.join(
        feature_dir, '%s_%d_%d.csv' % (team_level_file_, season - 1, season))

    data = pd.read_csv(results_file)
    standings = pd.read_csv(standings_file)
    f4teams = read_json(f4_file)

    # Specify the F4 teams of the *previous* year
    f4Teams = f4teams[str(season - 1)]

    # make game features
    feats = make_game_features(data, standings, f4Teams)

    # save features to file.
    logging.info('save match-level features')
    feats.to_csv(match_level_file, index=False)

    # make team features
    team_feats = make_team_features(data, standings, f4Teams)
    # save features to file
    logging.info('save team-level features')
    team_feats.to_csv(team_level_file, index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--season', required=True, type=int,
                        help="the ending year of the season")

    args = parser.parse_args()

    main(args.season)
