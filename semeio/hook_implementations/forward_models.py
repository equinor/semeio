import importlib
import os

from ert.shared.plugins.plugin_manager import hook_implementation
from ert.shared.plugins.plugin_response import plugin_response
from pkg_resources import resource_filename


def _remove_suffix(string, suffix):
    if not string.endswith(suffix):
        raise ValueError(f"{string} does not end with {suffix}")
    return string[: -len(suffix)]


def _get_forward_models_from_directory(directory):
    resource_directory = resource_filename("semeio", directory)

    all_files = [
        os.path.join(resource_directory, f)
        for f in os.listdir(resource_directory)
        if os.path.isfile(os.path.join(resource_directory, f))
    ]
    # ERT will look for an executable in the same folder as the forward model
    # configuration file is located. If the name of the config is the same as
    # the name of the executable, libres will be confused. The usual standard in
    # ERT would be to capitalize the config file. On OSX systems, which are
    # case-insensitive, this isn't viable. The config files are therefore
    # appended with "_CONFIG".  The forward models will be installed as FORWARD_MODEL_NAME,
    # and the FORWARD_MODEL_NAME_CONFIG will point to an executable named
    # forward_model_name - which we install with entry-points. The user can use the
    # forward model as normal:
    # FORWARD_MODEL FORWARD_MODEL_NAME()
    return {
        _remove_suffix(os.path.basename(path), "_CONFIG"): path for path in all_files
    }


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
