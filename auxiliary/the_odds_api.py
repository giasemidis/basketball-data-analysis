# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:07:16 2018

@author: Georgios
"""
import requests


def the_odds_api(api_key, sport='NBA', region='uk'):
    '''Returns odds of current games from https://the-odds-api.com/'''

    #api_key = 'f52acecbfb57955edc30cec0cd1ee035'

    url = 'https://api.the-odds-api.com/v2/odds/'
    parameters = {"sport": sport, "region": region,"apiKey": api_key}
    r = requests.get(url, params=parameters)
    if r.status_code != 200:
        print('Error, status code:', r.status_code)
        return None
    print(r.status_code)
    print(r.url)

    return r.json()