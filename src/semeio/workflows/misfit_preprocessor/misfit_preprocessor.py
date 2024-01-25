import logging

from ert import ErtScript
from ert.shared.plugins.plugin_manager import hook_implementation

logger = logging.getLogger(__name__)


class MisfitPreprocessorJob(ErtScript):
    # pylint: disable=method-hidden
    def run(self, *args, **kwargs):  # pylint: disable=unused-argument
        logger.info("Running misfit preprocessor")


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(MisfitPreprocessorJob, "MISFIT_PREPROCESSOR")
    workflow.description = "Misfit preprocessor has moved to ert"
    workflow.category = "observations.correlation"
