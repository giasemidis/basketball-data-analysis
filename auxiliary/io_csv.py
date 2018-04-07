# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 23:09:13 2018

@author: Georgios
"""

import csv
from . deco_path_valid import valid_file
from . deco_path_valid import valid_folder


@valid_folder
def write_csv(file, data, header=None):
    '''
    Writes data into csv file
    '''

    with open(file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        if type(header)==list and header:
            writer.writerow(header)
        writer.writerows(data)
    return


@valid_file
def read_csv(file, head=1):
    '''
    Reads data from csv file
    '''

    data = []
    header = []
    with open(file, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == head - 1:
                header = row
            else:
                data.append(row)
    return data, header