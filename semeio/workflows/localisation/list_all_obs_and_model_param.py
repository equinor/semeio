from res.enkf import ErtScript
from res.enkf.enums.enkf_var_type_enum import EnkfVarType
from semeio.workflows.localisation.local_script_lib import (
    get_param_from_ert,
    print_params,
)

debug_level = 1


def debug_print(text):
    if debug_level > 0:
        print(text)


def get_obs_from_ert(obs_data_from_ert_config):
    obs_node_names = []
    for obs_node in obs_data_from_ert_config:
        node_name = obs_node.key()
        obs_node_names.append(node_name)
    debug_print("\n")

    return obs_node_names


def get_param_node_name_list_from_ert(ert):
    ens_config = ert.ensembleConfig()
    keylist = ens_config.alloc_keylist()
    model_param_node_names = []
    for key in keylist:
        node = ens_config.getNode(key)
        variable_type = node.getVariableType()
        if variable_type == EnkfVarType.PARAMETER:
            model_param_node_names.append(key)
    return model_param_node_names


def get_result_node_name_list_from_ert(ert):
    ens_config = ert.ensembleConfig()
    keylist = ens_config.alloc_keylist()
    result_node_names = []
    for key in keylist:
        node = ens_config.getNode(key)
        variable_type = node.getVariableType()
        if variable_type == EnkfVarType.DYNAMIC_RESULT:
            result_node_names.append(key)
    return result_node_names


class LocalisationConfigJob(ErtScript):
    def run(self, *args):
        ert = self.ert()
        local_config = ert.getLocalConfig()

        # Clear all correlations
        local_config.clear()
        ert_obs = ert.getObservations()
        obs_from_ert_config = ert_obs.getAllActiveLocalObsdata()
        defined_ert_obs_names = get_obs_from_ert(obs_from_ert_config)
        if debug_level > 0:
            print(" -- Observation nodes:")
            for name in defined_ert_obs_names:
                print(f"     {name}")

        params_for_node, node_type, grid_config = get_param_from_ert(ert)
        print_params(params_for_node, node_type)
