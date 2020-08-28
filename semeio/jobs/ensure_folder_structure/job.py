import logging
import os


def _validate_path_element(path_elem):
    if os.path.isabs(path_elem):
        raise ValueError(
            "Configured structure cannot contain absolute elements."
            "Found: {}".format(path_elem)
        )
    if ".." in path_elem:
        raise ValueError(
            "Path elements should not contain '..'."
            "Found: {}".format(path_elem)
        )


def _expand_folders(folder_structure, root):
    if folder_structure is None:
        return [root]

    folder_list = []
    for elem, subs in folder_structure.items():
        _validate_path_element(elem)
        folder_list += _expand_folders(subs, os.path.join(root, elem))

    return folder_list


def ensure_folder_structure(folder_structure, root):
    root = os.path.realpath(root)
    folders = _expand_folders(folder_structure, root)
    for elem in folders:
        os.makedirs(elem, exist_ok=True)
        logging.info("Created folder: {}".format(elem))
