from ert_shared.libres_facade import LibresFacade
from ert_shared.plugins.plugin_manager import hook_implementation
from semeio.communication import SemeioScript


def print_obs_vector(vector):
    for node in vector:
        for index in range(len(node)):
            print(node.getStdScaling(index))


class PrintStd(
    SemeioScript
):  # pylint: disable=too-few-public-methods
    def run(self):
        obs = self.ert().getObservations()
        obs_vector = obs["POLY_OBS"]

        print_obs_vector(obs_vector)


@hook_implementation
def legacy_ertscript_workflow(config):
    config.add_workflow(PrintStd, "PRINT_STD")
