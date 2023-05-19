import os

from ert.shared.plugins.plugin_manager import ErtPluginManager

import semeio.hook_implementations.jobs
from semeio.workflows.ahm_analysis import ahmanalysis
from semeio.workflows.correlated_observations_scaling import cos
from semeio.workflows.csv_export2 import csv_export2
from semeio.workflows.localisation import local_config_script
from semeio.workflows.misfit_preprocessor import misfit_preprocessor
from semeio.workflows.spearman_correlation_job import spearman_correlation


def test_hook_implementations():
    plugin_manager = ErtPluginManager(
        plugins=[
            semeio.hook_implementations.jobs,
            spearman_correlation,
            misfit_preprocessor,
            cos,
            local_config_script,
            csv_export2,
            ahmanalysis,
        ]
    )

    expected_jobs = {
        "DESIGN_KW": "semeio/jobs/config_jobs/DESIGN_KW_CONFIG",
        "DESIGN2PARAMS": "semeio/jobs/config_jobs/DESIGN2PARAMS_CONFIG",
        "STEA": "semeio/jobs/config_jobs/STEA_CONFIG",
        "GENDATA_RFT": "semeio/jobs/config_jobs/GENDATA_RFT_CONFIG",
        "PYSCAL": "semeio/jobs/config_jobs/PYSCAL_CONFIG",
        "INSERT_NOSIM": "semeio/jobs/config_jobs/INSERT_NOSIM_CONFIG",
        "REMOVE_NOSIM": "semeio/jobs/config_jobs/REMOVE_NOSIM_CONFIG",
        "OTS": "semeio/jobs/config_jobs/OTS_CONFIG",
        "REPLACE_STRING": "semeio/jobs/config_jobs/REPLACE_STRING_CONFIG",
    }
    installable_jobs = plugin_manager.get_installable_jobs()
    for wf_name, wf_location in expected_jobs.items():
        assert wf_name in installable_jobs
        assert installable_jobs[wf_name].endswith(wf_location)

    assert set(installable_jobs.keys()) == set(expected_jobs.keys())

    expected_workflow_jobs = [
        "CORRELATED_OBSERVATIONS_SCALING",
        "SPEARMAN_CORRELATION",
        "CSV_EXPORT2",
        "MISFIT_PREPROCESSOR",
        "AHM_ANALYSIS",
        "LOCALISATION_JOB",
    ]
    installable_workflow_jobs = plugin_manager.get_installable_workflow_jobs()
    for wf_name, wf_location in installable_workflow_jobs.items():
        assert wf_name in expected_workflow_jobs
        assert os.path.isfile(wf_location)

    assert set(installable_workflow_jobs.keys()) == set(expected_workflow_jobs)


def test_hook_implementations_job_docs():
    plugin_manager = ErtPluginManager(plugins=[semeio.hook_implementations.jobs])

    installable_jobs = plugin_manager.get_installable_jobs()

    docs = plugin_manager.get_documentation_for_jobs()

    assert set(docs.keys()) == set(installable_jobs.keys())

    for job_name in installable_jobs.keys():
        assert docs[job_name]["description"] != ""
        assert docs[job_name]["category"] != "other"
