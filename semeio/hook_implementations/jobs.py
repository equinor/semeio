from ert_shared.plugins.plugin_manager import hook_implementation
from ert_shared.plugins.plugin_response import plugin_response


@hook_implementation
@plugin_response(plugin_name="semeio")
def installable_jobs():
    return {}


@hook_implementation
@plugin_response(plugin_name="semeio")
def installable_workflow_jobs():
    return {}
