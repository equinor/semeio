# pylint: disable=missing-docstring

import os
import semeio.hook_implementations.jobs

from semeio.workflows.misfit_preprocessor import misfit_preprocessor
from semeio.workflows.spearman_correlation_job import spearman_correlation
from semeio.workflows.correlated_observations_scaling import cos
from semeio.workflows.csv_export2 import csv_export2
from semeio.workflows.ahm_analysis import ahmanalysis
from ert_shared.plugins.plugin_manager import ErtPluginManager


def test_hook_implementations():
    pm = ErtPluginManager(
        plugins=[
            semeio.hook_implementations.jobs,
            spearman_correlation,
            misfit_preprocessor,
            cos,
            csv_export2,
            ahmanalysis,
        ]
    )

    expected_jobs = {
        "DESIGN_KW": "semeio/jobs/config_jobs/DESIGN_KW",
        "DESIGN2PARAMS": "semeio/jobs/config_jobs/DESIGN2PARAMS",
        "STEA": "semeio/jobs/config_jobs/STEA",
        "GENDATA_RFT": "semeio/jobs/config_jobs/GENDATA_RFT",
        "PYSCAL": "semeio/jobs/config_jobs/PYSCAL",
        "INSERT_NOSIM": "semeio/jobs/config_jobs/INSERT_NOSIM",
        "REMOVE_NOSIM": "semeio/jobs/config_jobs/REMOVE_NOSIM",
        "OTS": "semeio/jobs/config_jobs/OTS",
    }
    installable_jobs = pm.get_installable_jobs()
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
    ]
    installable_workflow_jobs = pm.get_installable_workflow_jobs()
    for wf_name, wf_location in installable_workflow_jobs.items():
        assert wf_name in expected_workflow_jobs
        assert os.path.isfile(wf_location)

    assert set(installable_workflow_jobs.keys()) == set(expected_workflow_jobs)


def test_hook_implementations_job_docs():
    pm = ErtPluginManager(plugins=[semeio.hook_implementations.jobs])

    installable_jobs = pm.get_installable_jobs()

    docs = pm.get_documentation_for_jobs()

    assert set(docs.keys()) == set(installable_jobs.keys())

    for job_name in installable_jobs.keys():
        assert docs[job_name]["description"] != ""
        assert docs[job_name]["category"] != "other"
