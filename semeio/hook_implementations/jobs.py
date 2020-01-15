import importlib
import os
from pkg_resources import resource_filename

from ert_shared.plugins.plugin_manager import hook_implementation
from ert_shared.plugins.plugin_response import plugin_response


def _get_jobs_from_directory(directory):
    resource_directory = resource_filename("semeio", directory)

    all_files = [
        os.path.join(resource_directory, f)
        for f in os.listdir(resource_directory)
        if os.path.isfile(os.path.join(resource_directory, f))
    ]
    return {os.path.basename(path): path for path in all_files}


# pylint: disable=no-value-for-parameter
@hook_implementation
@plugin_response(plugin_name="semeio")  # pylint: disable=no-value-for-parameter
def installable_jobs():
    return _get_jobs_from_directory("jobs/config_jobs")


def _get_module_variable_if_exists(module_name, variable_name, default=""):
    try:
        script_module = importlib.import_module(module_name)
    except ImportError:
        return default

    return getattr(script_module, variable_name, default)


@hook_implementation
@plugin_response(plugin_name="semeio")  # pylint: disable=no-value-for-parameter
def job_documentation(job_name):
    semeio_jobs = set(installable_jobs().data.keys())
    if job_name not in semeio_jobs:
        return None

    if job_name.endswith("NOSIM"):
        verb = job_name.split("_")[0].lower()
        insert = verb == "insert"
        return {
            "description": (
                "{} NOSIM {} the ECLIPSE data file. "
                "This will {} simulation in ECLIPSE."
            ).format(
                verb.capitalize(),
                "into" if insert else "from",
                "disable" if insert else "enable",
            ),
            "examples": "",
            "category": "utility.eclipse",
        }

    if job_name == "STEA" or job_name == "PYSCAL":
        module_name = "semeio.jobs.scripts.fm_{}".format(job_name.lower())
    elif job_name == "OTS":
        module_name = "semeio.jobs.scripts.overburden_timeshift"
    else:
        module_name = "semeio.jobs.scripts.{}".format(job_name.lower())

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
