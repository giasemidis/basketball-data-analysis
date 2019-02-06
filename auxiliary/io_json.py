# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:57:48 2018

@author: Georgios
"""

import json
from . deco_path_valid import valid_file
from . deco_path_valid import valid_folder


@valid_folder
def write_json(file, data):
    '''
    Writes data into json file
    '''
    with open(file, 'w', encoding='utf8') as outfile:
        json.dump(data, outfile, ensure_ascii=False)
    return


@valid_file
def read_json(file):
    '''
    Reads data from json file
    '''
    with open(file, 'r', encoding='utf8') as outfile:
        data = json.load(outfile)
    return data