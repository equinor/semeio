import logging
import shlex
import re


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
                (
                    "{} not found in design matrix, but this is probably a Perl file"
                ).format(unmatched)
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
    """Parses a list of strings, looking for key-value pairs pr. line
    separated by whitespace, into a dictionary.

    Spaces in keys and/or values are supported if quoted. Quotes
    in keys/values are not supported.

    Args:
        parameters (list of str)

    Returns:
        dict, with the keys and values parsed.

    Raises:
        ValueError, with error messages and all unparseable lines.
    """
    res = {}
    errors = []
    for line in parameters:
        line_parts = shlex.split(line)
        if not line_parts:
            continue
        if len(line_parts) == 1:
            errors += ["No value found in line {}".format(line)]
            continue
        if len(line_parts) > 2:
            errors += ["Too many values found in line {}".format(line)]
            continue
        key, value = line_parts
        if key in res:
            errors += ["{} is defined multiple times".format(key)]
            continue
        res[key] = value
    if errors:
        raise ValueError("\n".join(errors))
    return res
