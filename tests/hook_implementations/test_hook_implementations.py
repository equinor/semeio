import os

from ert.shared.plugins.plugin_manager import ErtPluginManager

import semeio.hook_implementations.forward_models
from semeio.workflows.ahm_analysis import ahmanalysis
from semeio.workflows.csv_export2 import csv_export2
from semeio.workflows.localisation import local_config_script


def test_hook_implementations():
    plugin_manager = ErtPluginManager(
        plugins=[
            semeio.hook_implementations.forward_models,
            local_config_script,
            csv_export2,
            ahmanalysis,
        ]
    )

    _fm_path = "semeio/forward_models/config"
    expected_forward_models = {
        forward_model_name: f"{_fm_path}/{forward_model_name}_CONFIG"
        for forward_model_name in [
            "DESIGN_KW",
            "DESIGN2PARAMS",
            "GENDATA_RFT",
            "PYSCAL",
            "INSERT_NOSIM",
            "REMOVE_NOSIM",
            "OTS",
            "REPLACE_STRING",
        ]
    }
    installable_fms = plugin_manager.get_installable_jobs()
    for fm_name, fm_location in expected_forward_models.items():
        assert fm_name in installable_fms
        assert installable_fms[fm_name].endswith(fm_location)

    assert set(installable_fms.keys()) == set(expected_forward_models.keys())

    expected_workflow_jobs = [
        "CSV_EXPORT2",
        "AHM_ANALYSIS",
        "LOCALISATION_JOB",
    ]
    installable_workflow_jobs = plugin_manager.get_installable_workflow_jobs()
    for wf_name, wf_location in installable_workflow_jobs.items():
        assert wf_name in expected_workflow_jobs
        assert os.path.isfile(wf_location)

    assert set(installable_workflow_jobs.keys()) == set(expected_workflow_jobs)


def test_hook_implementations_forward_model_docs():
    plugin_manager = ErtPluginManager(
        plugins=[semeio.hook_implementations.forward_models]
    )

    installable_fms = plugin_manager.get_installable_jobs()

    docs = plugin_manager.get_documentation_for_jobs()

    assert set(docs.keys()) == set(installable_fms.keys())

    for forward_model_name in installable_fms.keys():
        assert docs[forward_model_name]["description"] != ""
        assert docs[forward_model_name]["category"] != "other"
