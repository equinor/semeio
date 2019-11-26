import unittest
import semeio.hook_implementations.jobs
from ert_shared.plugins.plugin_manager import ErtPluginManager


class Test(unittest.TestCase):
    def test_hook_implementations(self):
        pm = ErtPluginManager(plugins=[semeio.hook_implementations.jobs])
        self.assertDictEqual({}, pm.get_installable_jobs())
        self.assertDictEqual({}, pm.get_installable_workflow_jobs())
