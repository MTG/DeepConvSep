'''Snazzy logging utils from http://github.com/bsilvert/utcondor.

Most of the code in this module was written by Bryan Silverthorn.
'''

curses = None
try:
    import curses
except ImportError:
    pass

import logging
import sys

__all__ = [
    'enable_default_logging',
    'get_logger',
]


class TTY_Formatter(logging.Formatter):
    '''A log formatter for console output.'''

    _DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    _COLORS = dict(
        TIME='\x1b[00m',
        PROC='\x1b[31m',
        NAME='\x1b[36m',
        LINE='\x1b[32m',
        END='\x1b[00m',
        )

    def __init__(self, stream=None, process_names=False):
        '''Construct this formatter.

        Provides colored output if the stream parameter is specified and is an
        acceptable TTY. We print hardwired escape sequences, which will probably
        break in some circumstances; for this unfortunate shortcoming, we
        apologize.
        '''
        colors = {k: '' for k in TTY_Formatter._COLORS}
        if stream and hasattr(stream, 'isatty') and stream.isatty() and curses:
            curses.setupterm()
            if curses.tigetnum('colors') > 2:
                colors = TTY_Formatter._COLORS
        parts = [
            '%%(levelname).1s ',
            '%(TIME)s%%(asctime)s%(END)s ',
            '%(PROC)s%%(processName)s%(END)s ' if process_names else '',
            '%(NAME)s%%(name)s%(END)s:',
            '%(LINE)s%%(lineno)d%(END)s ',
            '%%(message)s',
            ]
        logging.Formatter.__init__(self, ''.join(parts) % colors, TTY_Formatter._DATE_FORMAT)


def get_logger(name=None, level=None, default_level='INFO'):
    '''Get or create a logger.'''
    logger = logging.getLogger(name) if name else logging.root

    # python 3 changed the name of the dictionary that maps level names to level
    # numbers. this should let us use either name.
    names = {}
    for attr in ('_levelNames', '_nameToLevel'):
        try:
            names.update(getattr(logging, attr))
        except AttributeError:
            pass

    # set the default level, if the logger is new
    try:
        clean = logger.is_squeaky_clean
    except AttributeError:
        pass
    else:
        if clean and default_level is not None:
            logger.setLevel(names.get(default_level, default_level))

    # unconditionally set the logger level, if requested
    if level is not None:
        logger.setLevel(names.get(level, level))
        logger.is_squeaky_clean = False

    return logger


_default_logging_enabled = False

def enable_default_logging(default_level='INFO',
                           stream=sys.stdout,
                           process_names=False):
    '''Set up logging in the typical way.

    Parameters
    ----------
    default_level : str, optional
        Logging level. Defaults to INFO.
    stream : file-like, optional
        Stream for logging output. Defaults to ``sys.stdout``.
    process_names : bool, optional
        If True, include process names in logging output. Defaults to False.
    '''
    global _default_logging_enabled
    if _default_logging_enabled:
        return
    get_logger(level=default_level)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(TTY_Formatter(stream, process_names))
    logging.root.addHandler(handler)
    _default_logging_enabled = True
    return handler
