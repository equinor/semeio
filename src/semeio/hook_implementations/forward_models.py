import importlib
from typing import Dict

import importlib_resources
from ert import hook_implementation, plugin_response

from semeio.forward_models import (
    OTS,
    Design2Params,
    DesignKW,
    GenDataRFT,
    InsertNoSim,
    Pyscal,
    RemoveNoSim,
    ReplaceString,
)


def _remove_suffix(string: str, suffix: str) -> str:
    if not string.endswith(suffix):
        raise ValueError(f"{string} does not end with {suffix}")
    return string[: -len(suffix)]


def _get_forward_models_from_directory(directory: str) -> Dict[str, str]:
    resource_directory_ref = importlib_resources.files("semeio") / directory

    all_files = []
    with importlib_resources.as_file(resource_directory_ref) as resource_directory:
        all_files = [
            resource_directory / file
            for file in resource_directory.glob("*")
            if (resource_directory / file).is_file()
        ]

    # ERT will look for an executable in the same folder as the forward model
    # configuration file is located. If the name of the config is the same as
    # the name of the executable, ERT will be confused. The usual standard in
    # ERT would be to capitalize the config file. On OSX systems, which are
    # case-insensitive, this isn't viable. The config files are therefore
    # appended with "_CONFIG".
    # The forward models will be installed as FORWARD_MODEL_NAME,
    # and the FORWARD_MODEL_NAME_CONFIG will point to an executable named
    # forward_model_name - which we install with entry-points. The user can use the
    # forward model as normal:
    # FORWARD_MODEL FORWARD_MODEL_NAME()
    return {_remove_suffix(path.name, "_CONFIG"): str(path) for path in all_files}


@hook_implementation
@plugin_response(plugin_name="semeio")
def installable_jobs():
    return _get_forward_models_from_directory("forward_models/config")


def _get_module_variable_if_exists(module_name, variable_name, default=""):
    try:
        script_module = importlib.import_module(module_name)
    except ImportError:
        return default

    return getattr(script_module, variable_name, default)


@hook_implementation
@plugin_response(plugin_name="semeio")
def job_documentation(job_name):
    forward_model_name = job_name
    semeio_forward_models = set(installable_jobs().data.keys())
    if forward_model_name not in semeio_forward_models:
        return None

    if forward_model_name.endswith("NOSIM"):
        verb = forward_model_name.split("_")[0].lower()
        insert = verb == "insert"
        return {
            "description": (
                f"{verb.capitalize()} NOSIM "
                f"{'into' if insert else 'from'} the ECLIPSE data file. "
                f"This will {'disable' if insert else 'enable'} simulation in ECLIPSE. "
                "NB: the forward model does not currently work on osx systems"
            ),
            "examples": "",
            "category": "utility.eclipse",
        }

    if forward_model_name == "PYSCAL":
        module_name = f"semeio.forward_models.scripts.fm_{forward_model_name.lower()}"
    elif forward_model_name == "OTS":
        module_name = "semeio.forward_models.scripts.overburden_timeshift"
    else:
        module_name = f"semeio.forward_models.scripts.{forward_model_name.lower()}"

    description = _get_module_variable_if_exists(
        module_name=module_name, variable_name="description"
    )
    examples = _get_module_variable_if_exists(
        module_name=module_name, variable_name="examples"
    )
    category = _get_module_variable_if_exists(
        module_name=module_name, variable_name="category", default="other"
    )

    return {
        "description": description,
        "examples": examples,
        "category": category,
    }


@hook_implementation
@plugin_response(plugin_name="semeio")
def installable_forward_model_steps():
    return [
        Design2Params,
        DesignKW,
        GenDataRFT,
        OTS,
        Pyscal,
        InsertNoSim,
        RemoveNoSim,
        ReplaceString,
    ]
