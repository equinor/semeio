import unittest
import os
import subprocess

from ert_statoil.testcase import TestCase

try:
  from ecl.util.test import TestAreaContext
  from ecl.util.util import BoolVector
except ImportError:
  from ert.test import TestAreaContext
  from ert.util import BoolVector

from res.test import ErtTestContext
from res.enkf import EnkfFs, EnkfConfigNode, NodeId, EnkfNode
from res.enkf import EnKFMain, ErtRunContext
from res.enkf.enums import EnKFFSType
from res.enkf.enums import RealizationStateEnum

from res.util import ResVersion
from unittest import skipIf

currentVersion = ResVersion( )

class NexusTest(TestCase):

    def test_fail(self):
       executable = os.path.join(self.ETC_PATH, "ERT/Config/jobs/ressim/Scripts/ert_nexus.py")

       #ert_nexus.py assumes a perfect setup of configuration file SPE1.fcs while running nexus-specific execution files
       #However, SPE1.fcs points to include files in a nexus_data directory, if this is missing nexus fails.
       #If nexus fails, ert nexus.py shall also fail.
       with TestAreaContext("nexus_shall_fail") as work_area:
         source_file = os.path.join(self.PROJECT_ROOT, 'testdata/nexus_SPE1/SPE1.fcs')
         work_area.copy_file(source_file)
         with self.assertRaises(Exception):
           subprocess.check_call([executable, "SPE1", "ECL_SPE1"])



    @skipIf(currentVersion < (2,3,"X") , "Must have at least version 2.3")
    @skipIf(True, "Test is temporarily skipped after site-config file became template") 
    def test_SPE1(self):

       os.environ["ERT_SITE_CONFIG"] = os.path.join(self.ETC_PATH, "ERT/site-config")
       with ErtTestContext( "nexus_spe1", model_config = os.path.join(self.PROJECT_ROOT, 'testdata/nexus_SPE1/nexus.ert'), store_area = True) as ctx:
         ert = ctx.getErt( )
         fs_manager = ert.getEnkfFsManager()
         result_fs = fs_manager.getCurrentFileSystem( )

         model_config = ert.getModelConfig( )
         runpath_fmt = model_config.getRunpathFormat( )
         jobname_fmt = model_config.getJobnameFormat( )
         subst_list = ert.getDataKW( )
         itr = 0
         mask = BoolVector( default_value = True, initial_size = 1 )
         run_context = ErtRunContext.ensemble_experiment( result_fs, mask, runpath_fmt, jobname_fmt, subst_list, itr)
         ert.getEnkfSimulationRunner().createRunPath( run_context )

         job_queue = ert.get_queue_config().create_job_queue()
         num = ert.getEnkfSimulationRunner().runEnsembleExperiment(job_queue, run_context)
         assert(num == 1)





if __name__ == "__main__":
  unittest.main()
