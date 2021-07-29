# Localisation configuration script
# Written by: O.Lia
# Date: Mayl 2021
# Current version:
#   Localisation configuration file in YAML format is read for
#   parameters and this script setup the localisation in ERT.

import semeio.workflows.localisation.local_script_lib as local

from ert_shared.plugins.plugin_manager import hook_implementation
from semeio.communication import SemeioScript
from semeio.workflows.localisation.localisation_config import LocalisationConfig


class LocalisationConfigJob(SemeioScript):
    def run(self, *args):
        ert = self.ert()

        # Clear all correlations
        local.clear_correlations(ert)

        # Read yml file with specifications
        config_dict = local.read_localisation_config(args)

        # Get all observations from ert instance
        ert_obs_list = local.get_observations_from_ert(ert)

        ert_param_dict, ert_node_type_dict = local.get_param_from_ert(ert)

        config = LocalisationConfig(
            observations=ert_obs_list,
            parameters=ert_param_dict,
            node_type=ert_node_type_dict,
            **config_dict,
        )

        local.add_ministeps(config, ert_param_dict, ert)
        print("\nFinished localisation setup.")


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(LocalisationConfigJob, "LOCALISATION_JOB")
    workflow.description = "test"
    workflow.category = "observations.correlation"
