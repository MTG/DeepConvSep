'''Useful utilities for command-line arguments.'''

import argparse

__all__ = [
    'add_arg',
    'add_command',
    'add_group',
    'add_mutex',
    'parse_args',
    'parse_known_args',
    'print_usage',
    'print_help',
]

class Parser(argparse.ArgumentParser):
    '''This class provides some sane default command-line argument behaviors.

    In particular, the help formatter includes default values, and arguments can
    be loaded from files by using the "@" prefix.

    Files can contain arguments in two ways: one per line, or many-per-line. For
    one-per-line arguments, spaces etc. are preserved, while for many-per-line
    arguments, the line must start with a dash, and multiple arguments are split
    on whitespace. In all cases, shell-style comments are removed from the file
    before processing; escape "#" characters (e.g., if you're trying to give a
    hexadecimal color string) using a backslash.
    '''

    SANE_DEFAULTS = dict(
        fromfile_prefix_chars='@',
        conflict_handler='resolve',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def __init__(self, *args, **kwargs):
        kw = {}
        kw.update(Parser.SANE_DEFAULTS)
        kw.update(kwargs)
        super(Parser, self).__init__(*args, **kw)
        self._subparsers = None

    def convert_arg_line_to_args(self, line):
        '''Remove # comments and blank lines from arg files.'''
        S = 'eF5P86904hSOXAn9YLA1JMNXxG8EX7DA'
        line = line.replace(r'\#', S).split('#')[0].strip().replace(S, '#')
        if line:
            if line[0] == '-' and ' ' in line:
                for p in line.split():
                    yield p
            else:
                yield line

PARSER = None

def _parser():
    global PARSER
    if PARSER is None:
        PARSER = Parser()
    return PARSER

def add_mutex(*args, **kwargs):
    '''Add a mutually-exclusive argparse group.

    Returns
    -------
    A mutually-exclusive argparse argument group object.
    '''
    return _parser().add_mutually_exclusive_group(*args, **kwargs)


def add_group(*args, **kwargs):
    '''Add an argparse argument group.

    Returns
    -------
    An argparse argument group object.
    '''
    return _parser().add_argument_group(*args, **kwargs)


def add_arg(*args, **kwargs):
    '''Add an argparse argument.'''
    return _parser().add_argument(*args, **kwargs)


def add_command(*args, **kwargs):
    '''Add an argparse command parser.

    The name of the command will be stored in the "command_name" argparse
    variable.

    Returns
    -------
    An argparse command parser object.
    '''
    parser = _parser()
    if parser._subparsers is None:
        parser._subparsers = parser.add_subparsers(dest='command_name')
    return parser._subparsers.add_parser(*args, **kwargs)


def parse_args(**overrides):
    '''Parse command-line arguments, overriding with keyword arguments.

    Returns
    -------
    args : namespace
        The command-line argument namespace object.
    '''
    args = _parser().parse_args()
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def parse_known_args(**overrides):
    '''Parse known command-line arguments, overriding with keyword arguments.

    Returns
    -------
    args : namespace
        The command-line argument namespace object.
    rest : list of strings
        A list containing unknown command-line arguments.
    '''
    args, rest = _parser().parse_known_args()
    for k, v in overrides.items():
        setattr(args, k, v)
    return args, rest


def print_usage(*args, **kwargs):
    '''Print usage information for the currently defined parser.

    All positional and keyword arguments are passed to the underlying parser.
    '''
    _parser().print_usage(*args, **kwargs)


def print_help(*args, **kwargs):
    '''Print help information for the currently defined parser.

    All positional and keyword arguments are passed to the underlying parser.
    '''
    _parser().print_help(*args, **kwargs)
