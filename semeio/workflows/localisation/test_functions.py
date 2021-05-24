# import yaml
# import pathlib

# from functools import partial

# import cwrap
from res.enkf import ErtScript

# from res.enkf.enums.ert_impl_type_enum import ErtImplType
# from res.enkf.enums.active_mode_enum import ActiveMode

# from res.enkf import EnkfObs
# from ecl.grid import EclRegion, EclGrid
# from ecl.eclfile import EclKW, Ecl3DKW
# from ecl.ecl_type import EclDataType
# from ecl.util.util import IntVector

from semeio.workflows.localisation import local_script_lib as local

# Global variables
debug_level = 1

# Dictionary with key =(obs_name, model_param_name) containing value
# True or False depending on whether the correlation is active or not.
# Use this to check that no double specification of a correlation appears.
correlation_table = {}


class LocalisationConfigJob(ErtScript):
    def run(self, *args):
        ert = self.ert()

        # Read yml file with specifications
        all_kw = local.read_localisation_config(args)

        # Read observation groups into the obs_groups dictionary
        obs_groups = local.read_obs_groups(ert, all_kw)
        print(f"obs_groups: {obs_groups}")
