# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 23:08:36 2018

@author: Georgios
"""

import os.path
import sys


def valid_file(func):
    """Checks the validity of the input file of read-data fuction. If files
    does not exist, it exists.
    """

    def wrapper(filename, *args, **kwargs):
        if os.path.isfile(filename):
            a = func(filename, *args, **kwargs)
            return a
        else:
            sys.exit('File %s not found' % filename)
            return
    return wrapper


def valid_folder(func):
    """Checks the validity of the output directory of a write-data function. If
    directory does not exist, it exists.
    """

    def wrapper(filepath, *args, **kwargs):
        if os.path.isdir(os.path.dirname(filepath)):
            a = func(filepath, *args, **kwargs)
            return a
        else:
            sys.exit('Directory %s not found' % filepath)
            return
    return wrapper