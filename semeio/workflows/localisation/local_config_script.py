# Localisation configuration script
# Written by: O.Lia
# Date: Mayl 2021
# Current version:
#   Localisation configuration file in YAML format is read for
#   parameters and this script setup the localisation in ERT.
# from functools import partial
# import cwrap
from res.enkf import ErtScript
from res.enkf.enums.ert_impl_type_enum import ErtImplType
import semeio.workflows.localisation.local_script_lib as local

# from res.enkf import EnkfObs
# from ecl.grid import EclRegion, EclGrid
# from ecl.eclfile import EclKW, Ecl3DKW
# from ecl.ecl_type import EclDataType
# from ecl.util.util import IntVector
from ert_shared.plugins.plugin_manager import hook_implementation


class LocalisationConfigJob(ErtScript):
    def run(self, *args):
        ert = self.ert()

        # Read yml file with specifications
        all_kw = local.read_localisation_config(args)

        # Path to configuration setup
        #        config_path = all_kw["config_path"]

        # Get all observations from ert instance
        ert_obs_list = local.get_observations_from_ert(ert)

        # Get dict of observation groups from main keyword 'obs_groups'
        obs_groups = local.read_obs_groups(ert_obs_list, all_kw)

        # Get all parameters nodes with parameters defined by GEN_KW from ert instance
        ert_param_dict = local.get_param_from_ert(ert, impl_type=ErtImplType.GEN_KW)
        local.debug_print(
            f"-- All specified scalar parameters from ERT instance:\n"
            f"{ert_param_dict}\n"
        )

        # Get dict of user defined model parameter groups from
        # main keyword 'model_param_groups'
        param_groups = local.read_param_groups(ert_param_dict, all_kw)

        # Get specification of active correlations for localisation

        correlation_specification = local.read_correlation_specification(
            all_kw,
            ert_obs_list,
            ert_param_dict,
            obs_groups=obs_groups,
            param_groups=param_groups,
        )
        local.debug_print(f" -- Correlation specification: {correlation_specification}")
        number_of_duplicates = local.check_for_duplicated_correlation_specifications(
            correlation_specification
        )
        local.debug_print(
            f" -- Number of duplicated correlations: {number_of_duplicates}"
        )
        if number_of_duplicates > 0:
            raise ValueError(
                f"Number of duplicated correlations specified is: "
                f"{number_of_duplicates}"
            )


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(LocalisationConfigJob, "LOCALISATION_JOB")
    workflow.description = "test"
    workflow.category = "observations.correlation"
