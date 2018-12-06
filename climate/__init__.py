'''Some utilities for command line interfaces!'''

from .log import *
from .flags import *

import plac
import sys

logging = get_logger(__name__)

__version__ = '0.4.6'


def annotate(*args, **kwargs):
    '''Return a decorator for plac-style argument annotations.'''
    return plac.annotations(*args, **kwargs)


def call(main, default_level='INFO', stream=sys.stdout, process_names=False):
    '''Enable logging and start up a main method.

    Parameters
    ----------
    main : callable
        The main method to invoke after initialization.
    default_level : str, optional
        Logging level. Defaults to INFO.
    stream : file-like, optional
        Stream for logging output. Defaults to ``sys.stdout``.
    process_names : bool, optional
        If True, include process names in logging output. Defaults to False.
    '''
    enable_default_logging(
        default_level=default_level,
        stream=stream,
        process_names=process_names)
    from . import flags
    if flags.PARSER is None:
        return plac.call(main)
    args, rest = parse_known_args()
    if rest:
        logging.debug('unknown arguments: %s', rest)
    logging.debug('running with arguments:')
    kwargs = vars(args)
    for k in sorted(kwargs):
        logging.debug('--%s = %s', k, kwargs[k])
    return main(args)
