import os


def is_valid_parent_path(parser, x):
    parent_path = '.' if os.path.split(x)[0] == '' else os.path.split(x)[0]
    if not os.path.isdir(parent_path):
        parser.error('Parent path %s of output file not valid.' % parent_path)
    else:
        return str(x)
