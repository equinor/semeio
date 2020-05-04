def has_keys(main_list, sub_list, error_msg):
    """
    Checks that all sub_list are present and returns a list of error messages
    """
    return [error_msg.format(key) for key in sub_list if key not in main_list]


def is_subset(main_list, sub_list):
    """
    Checks if all the keys in sub_list are present in main_list and returns list of
    error messages
    """
    error_msg = "Update key: {} missing from calculate keys: {}"
    missing_keys = set(sub_list).difference(set(main_list))
    return [error_msg.format(missing_key, main_list) for missing_key in missing_keys]
