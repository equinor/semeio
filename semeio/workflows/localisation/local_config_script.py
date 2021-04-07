# Localisation configuration script
# Written by: O.Lia
# Date: Mayl 2021
# Current version:
#   Localisation configuration file in YAML format is read for
#   parameters and this script setup the localisation in ERT.
from ert_shared.libres_facade import LibresFacade
from ert_shared.plugins.plugin_manager import hook_implementation

import semeio.workflows.localisation.local_script_lib as local
from semeio.communication import SemeioScript
from semeio.workflows.localisation.localisation_config import LocalisationConfig


class LocalisationConfigJob(SemeioScript):
    def run(self, *args):
        ert = self.ert()
        facade = LibresFacade(self.ert())
        # Clear all correlations
        local.clear_correlations(ert)

        # Read yml file with specifications
        config_dict = local.read_localisation_config(args)

        # Get all observations from ert instance
        obs_keys = [
            facade.get_observation_key(nr)
            for nr, _ in enumerate(facade.get_observations())
        ]

        ert_parameters = local.get_param_from_ert(ert.ensembleConfig())

        config = LocalisationConfig(
            observations=obs_keys,
            parameters=ert_parameters.to_list(),
            **config_dict,
        )

        local.add_ministeps(
            config,
            ert_parameters.to_dict(),
            ert.getLocalConfig(),
            ert.ensembleConfig(),
            ert.eclConfig().getGrid(),
        )


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(LocalisationConfigJob, "LOCALISATION_JOB")
    workflow.description = "test"
    workflow.category = "observations.correlation"
