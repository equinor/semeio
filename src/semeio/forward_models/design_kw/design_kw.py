import logging
import re
import shlex
from collections.abc import Mapping

from ert import ForwardModelStepWarning

_STATUS_FILE_NAME = "DESIGN_KW.OK"

_logger = logging.getLogger(__name__)


class ParameterSyntaxError(ValueError):
    pass


def run(
    template_file_name: str,
    result_file_name: str,
    log_level: int,
    parameters_file_name: str = "parameters.txt",
) -> bool:
    # Get all key, value pairs
    # If FWL key is having multiple entries in the parameters file
    # KeyError is raised. This will be logged, and no OK
    # file is written

    _logger.setLevel(log_level)

    valid = True

    with open(parameters_file_name, encoding="utf-8") as parameters_file:
        parameters = parameters_file.readlines()

    key_vals = extract_key_value(parameters)

    key_vals.update(rm_genkw_prefix(key_vals))

    with open(template_file_name, encoding="utf-8") as template_file:
        template = template_file.readlines()

    with open(result_file_name, "w", encoding="utf-8") as result_file:
        for line in template:
            if not is_comment(line):
                for key, value in key_vals.items():
                    line = line.replace(f"<{key}>", str(value))

                if find_matching_errors(line, template_file_name, template):
                    valid = False

            result_file.write(line)

    if valid:
        with open(_STATUS_FILE_NAME, "w", encoding="utf-8") as status_file:
            status_file.write("DESIGN_KW OK\n")

    return valid


def find_matching_errors(
    line: str, template_file_name: str, template: list[str]
) -> list[str]:
    errors = []
    for unmatched in unmatched_templates(line):
        if is_perl(template_file_name, template):
            _logger.warning(
                f"{unmatched} not found in design matrix, "
                f"but this is probably a Perl file"
            )
        elif is_xml(template_file_name, template):
            _logger.warning(
                f"{unmatched} not found in design matrix, "
                f"but this is probably an xml file"
            )
        else:
            error_msg = f"{unmatched} not found in design matrix"
            _logger.error(error_msg)
            errors.append(error_msg)
    return errors


def is_perl(file_name, template):
    return file_name.endswith(".pl") or template[0].find("perl") != -1


def is_xml(file_name: str, template: list[str]) -> bool:
    return file_name.endswith(".xml") or template[0].find("?xml") != -1


def unmatched_templates(line):
    bracketpattern = re.compile("<.+?>")
    if bracketpattern.search(line):
        return bracketpattern.findall(line)
    return []


def is_comment(line):
    ecl_comment_pattern = re.compile("^--")
    std_comment_pattern = re.compile("^#")
    return ecl_comment_pattern.search(line) or std_comment_pattern.search(line)


def extract_key_value(parameters: list[str]) -> dict[str, str]:
    """Parses a list of strings, looking for key-value pairs pr. line
    separated by whitespace, into a dictionary.

    Spaces in keys and/or values are supported if quoted. Quotes
    in keys/values are not supported.

    Args:
        parameters (list of str)

    Returns:
        dict, with the keys and values parsed.

    Raises:
        ValueError, with error messages and all unparsable lines.
    """
    res = {}
    errors = []
    for line in parameters:
        try:
            line_parts = shlex.split(line)
        except ValueError as e:
            errors.append(f"Line '{line}' failed with '{e!s}'")
            continue
        if not line_parts:
            continue
        if len(line_parts) == 1:
            errors += [f"No value found in line {line}"]
            continue
        if len(line_parts) > 2:
            errors += [f"Too many values found in line {line}"]
            continue
        key, value = line_parts
        if key in res:
            errors += [f"{key} is defined multiple times"]
            continue
        res[key] = value
    if errors:
        raise ParameterSyntaxError("\n".join(errors))
    return res


def rm_genkw_prefix(paramsdict, ignoreprefixes="LOG10_"):
    """Strip prefixes from keys in a dictionary.

    Prefix is any string before a colon. No colon means no prefix.

    Only keys unique after prefix-stripping
    are included. For intentional duplicates, as when ERT
    prepares LOG10_ values, these are ignored by default in this
    function.

    Args:
        paramsdict (dict): Dictionary with parameter names as keys.
        ignoreprefixes (str or list of str): If any of these strings
            are found at the start of the prefix, they are removed
            from the dictionary before uniqueness is determined.

    Returns:
        Subset of the incoming dictionary (ignored keys are dropped), and with
        stripped prefixes from keys.
    """
    if ignoreprefixes is None:
        ignoreprefixes = []
    if isinstance(ignoreprefixes, str):
        ignoreprefixes = [ignoreprefixes]
    ignoreprefixes = filter(None, ignoreprefixes)

    for ignore_str in ignoreprefixes:
        paramsdict = {
            key: paramsdict[key]
            for key in paramsdict
            if ":" not in key or not key.startswith(ignore_str)
        }

    keyvalues = [
        (key.split(":")[1], value) if ":" in key else (key, value)
        for key, value in paramsdict.items()
    ]

    keys = [keyval[0] for keyval in keyvalues]

    duplicates = {keyvalue[0] for keyvalue in keyvalues if keys.count(keyvalue[0]) > 1}
    if duplicates:
        _logger.warning(f"Key(s) {list(duplicates)} can only be used with prefix.")

    return {
        keyvalue[0]: keyvalue[1]
        for keyvalue in keyvalues
        if keys.count(keyvalue[0]) == 1
    }


def validate_template_keys(
    key_vals: Mapping[str, str], template_file_name: str
) -> None:
    try:
        with open(template_file_name, encoding="utf-8") as template_file:
            template = template_file.readlines()
    except (OSError, UnicodeDecodeError) as err:
        ForwardModelStepWarning.warn(
            f"Could not read template file {template_file_name}: {err}"
        )
        return

    for line in template:
        if is_comment(line):
            continue
        for key, value in key_vals.items():
            line = line.replace(f"<{key}>", str(value))
            for error in find_matching_errors(line, template_file_name, template):
                ForwardModelStepWarning.warn(error)


def validate_configuration(
    template_file_name: str, parameters_file_name: str = "parameters.txt"
) -> None:
    try:
        with open(parameters_file_name, encoding="utf-8") as parameters_file:
            parameters = parameters_file.readlines()
    except (OSError, UnicodeDecodeError) as err:
        ForwardModelStepWarning.warn(
            f"Could not read parameters file {parameters_file_name}: {err}"
        )
        return
    try:
        key_vals = extract_key_value(parameters)
    except ParameterSyntaxError as err:
        ForwardModelStepWarning.warn(str(err))
        return
    key_vals.update(rm_genkw_prefix(key_vals))

    validate_template_keys(key_vals, template_file_name)
