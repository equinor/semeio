import importlib
import os
from pkg_resources import resource_filename

from ert_shared.plugins.plugin_manager import hook_implementation
from ert_shared.plugins.plugin_response import plugin_response


def _remove_suffix(string, suffix):
    if not string.endswith(suffix):
        raise ValueError(f"{string} does not end with {suffix}")
    return string[: -len(suffix)]


def _get_jobs_from_directory(directory):
    resource_directory = resource_filename("semeio", directory)

    all_files = [
        os.path.join(resource_directory, f)
        for f in os.listdir(resource_directory)
        if os.path.isfile(os.path.join(resource_directory, f))
    ]
    # libres will look for an executable in the same folder as the job configuration
    # file is located. If the name of the config is the same as the name of the executable,
    # libres will be confused. The usual standard in ERT would be to capitalize the config
    # file. On OSX systems, which are case-insensitive, this isn't viable. The job config
    # files are therefore appended with "_CONFIG".
    # The jobs will be installed as JOB_NAME JOB_NAME_CONFIG, and the JOB_NAME_CONFIG will
    # point to an executable named job_name - which we install with entry-points. The user
    # can use the forward model job as normal:
    # SIMULATION_JOB JOB_NAME
    return {
        _remove_suffix(os.path.basename(path), "_CONFIG"): path for path in all_files
    }


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
                "{} NOSIM {} the ECLIPSE data file. "  # pylint: disable=consider-using-f-string
                "This will {} simulation in ECLIPSE. NB: the job does not currently work on osx systems"
            ).format(
                verb.capitalize(),
                "into" if insert else "from",
                "disable" if insert else "enable",
            ),
            "examples": "",
            "category": "utility.eclipse",
        }

    if job_name == "STEA" or job_name == "PYSCAL":
        module_name = f"semeio.jobs.scripts.fm_{job_name.lower()}"
    elif job_name == "OTS":
        module_name = "semeio.jobs.scripts.overburden_timeshift"
    else:
        module_name = f"semeio.jobs.scripts.{job_name.lower()}"

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
