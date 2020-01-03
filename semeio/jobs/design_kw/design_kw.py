import json
import logging
import os
import re
import string
import sys

_STATUS_FILE_NAME = "DESIGN_KW.OK"

_logger = logging.getLogger(__name__)


def run(
    template_file_name,
    result_file_name,
    log_level,
    parameters_file_name="parameters.txt",
):
    # Get all key, value pairs
    # If FWL key is having multiple entries in the parameters file
    # KeyError is raised. This will be logged, and no OK
    # file is written

    _logger.setLevel(log_level)

    valid = True
    with open(parameters_file_name) as parameters_file:
        parameters = parameters_file.readlines()

    key_vals = extract_key_value(parameters)

    with open(template_file_name, "r") as template_file:
        template = template_file.readlines()

    if valid:
        with open(result_file_name, "w") as result_file:
            for line in template:
                if not is_comment(line):
                    for key, value in key_vals.items():
                        line = line.replace("<{}>".format(key), str(value))

                    if not all_matched(line, template_file_name, template):
                        valid = False

                result_file.write(line)

    if valid:
        with open(_STATUS_FILE_NAME, "w") as status_file:
            status_file.write("DESIGN_KW OK\n")


def all_matched(line, template_file_name, template):
    valid = True
    for unmatched in unmatched_templates(line):
        if is_perl(template_file_name, template):
            _logger.warn(
                "{} not found in design matrix, but this is probably a Perl file".format(
                    unmatched
                )
            )
        else:
            _logger.error("{} not found in design matrix".format(unmatched))
            valid = False
    return valid


def is_perl(file_name, template):
    return file_name.endswith(".pl") or template[0].find("perl") != -1


def unmatched_templates(line):
    bracketpattern = re.compile("<.+?>")
    if bracketpattern.search(line):
        return bracketpattern.findall(line)
    else:
        return []


def is_comment(line):
    ecl_comment_pattern = re.compile("^--")
    std_comment_pattern = re.compile("^#")
    return ecl_comment_pattern.search(line) or std_comment_pattern.search(line)


def extract_key_value(parameters):
    # Extract all key, value pairs from parameters file
    # If key already exists, raise Exception

    res = {}
    for line in parameters:
        key, value = line.split()
        if key in res:
            raise SystemExit(
                "{} is defined multiple times in parameters file".format(key)
            )
        res[key] = value
    return res
