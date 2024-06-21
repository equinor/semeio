import argparse
from pathlib import Path

description = """
Performs inplace string replacement in a files.

| ARGS
| FROM: string to match what to replace
| TO: expression to replace the match
| FILE: name of file to perform replacement in.

| Example:
| FORWARD_MODEL REPLACE_STRING(<FROM>=something, <TO>=else, <FILE>=file.txt)
|  > replace all something to else in file.txt

Due to ERT particular way of parsing config files, there are some curiosities
To escape characters like , use '', however this misses white space
To escape whitespace use ""

"""


def _get_args_parser():
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-o",
        "--original",
        help="Original text",
    )
    parser.add_argument(
        "-n",
        "--new",
        help="New text",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="File to where replacement should occur",
    )
    return parser


def main_entry_point():
    parser = _get_args_parser()
    options = parser.parse_args()
    file = Path(options.file)
    file.write_text(
        file.read_text(encoding="utf-8").replace(options.original, options.new),
        encoding="utf-8",
    )
