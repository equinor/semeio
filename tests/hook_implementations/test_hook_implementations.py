import sys
import unittest

import pytest

import semeio.hook_implementations.jobs
from ert_shared.plugins.plugin_manager import ErtPluginManager


@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3")
class Test(unittest.TestCase):
    def test_hook_implementations(self):
        pm = ErtPluginManager(plugins=[semeio.hook_implementations.jobs])

        expected_jobs = {
            "DESIGN_KW": "semeio/jobs/config_jobs/DESIGN_KW",
            "DESIGN2PARAMS": "semeio/jobs/config_jobs/DESIGN2PARAMS",
            "STEA": "semeio/jobs/config_jobs/STEA",
        }
        installable_jobs = pm.get_installable_jobs()
        for wf_name, wf_location in expected_jobs.items():
            self.assertTrue(wf_name in installable_jobs)
            self.assertTrue(installable_jobs[wf_name].endswith(wf_location))

        self.assertSetEqual(set(installable_jobs.keys()), set(expected_jobs.keys()))

        expected_workflow_jobs = {
            "CORRELATED_OBSERVATIONS_SCALING": "semeio/jobs/config_workflow_jobs/CORRELATED_OBSERVATIONS_SCALING",
            "SPEARMAN_CORRELATION": "semeio/jobs/config_workflow_jobs/SPEARMAN_CORRELATION",
        }
        installable_workflow_jobs = pm.get_installable_workflow_jobs()
        for wf_name, wf_location in expected_workflow_jobs.items():
            self.assertTrue(wf_name in installable_workflow_jobs)
            self.assertTrue(installable_workflow_jobs[wf_name].endswith(wf_location))

        self.assertSetEqual(
            set(installable_workflow_jobs.keys()), set(expected_workflow_jobs.keys())
        )
