import argparse
import logging
import sys
from pathlib import Path


def setup_logging():
    logger = logging.getLogger("semeio")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    std_out_handler = logging.StreamHandler(sys.stdout)
    std_out_handler.setFormatter(formatter)
    std_out_handler.setLevel(logging.INFO)
    logger.addHandler(std_out_handler)

    std_err_handler = logging.StreamHandler(sys.stderr)
    std_err_handler.setFormatter(formatter)
    std_err_handler.setLevel(logging.WARNING)
    logger.addHandler(std_err_handler)


def valid_file(arg):
    if Path(arg).is_file():
        return arg
    raise argparse.ArgumentTypeError(f"{arg} is not an existing file!")
