# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:13:40 2018

@author: Georgios
"""

import requests


def sports_radar_api(api_key, method):
    '''see https://developer.sportradar.com/io-docs'''

    #api_key = 'q6jmutjvbxwk7eg69ypu73fx'
    #example = 'http://api.sportradar.us/basketball-t1/en/schedules/2018-03-30/results.json?api_key=q6jmutjvbxwk7eg69ypu73fx'

    url = 'http://api.sportradar.us/basketball-t1/en/'
    url = url + method
    parameters = {'api_key': api_key}
    r = requests.get(url, params=parameters)
    if r.status_code != 200:
        print('Error, status code:', r.status_code)
        return None
    print(r.status_code)
    print(r.url)

    return r.json()