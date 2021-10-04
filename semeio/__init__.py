import argparse
import logging
import os
import sys


from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
std_out_handler = logging.StreamHandler(sys.stdout)
std_out_handler.setFormatter(formatter)
std_out_handler.setLevel(logging.INFO)
logger.addHandler(std_out_handler)

std_err_handler = logging.StreamHandler(sys.stderr)
std_err_handler.setFormatter(formatter)
std_err_handler.setLevel(logging.WARNING)
logger.addHandler(std_err_handler)


def valid_file(arg):
    if os.path.isfile(arg):
        return arg
    raise argparse.ArgumentTypeError(f"{arg} is not an existing file!")
