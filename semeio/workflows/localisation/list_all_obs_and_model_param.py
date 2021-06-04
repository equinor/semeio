from res.enkf import ErtScript
from res.enkf.enums.enkf_var_type_enum import EnkfVarType
from res.enkf.enums.ert_impl_type_enum import ErtImplType


debug_level = 1


def debug_print(text):
    if debug_level > 0:
        print(text)


def get_obs_names_for_node(obs_group, ert):
    ert_obs = ert.getObservations()
    obs_data_from_ert_config = ert_obs.getAllActiveLocalObsdata()
    obs_names = []
    for obs_object in obs_data_from_ert_config:
        if obs_object in obs_group:
            obs_names.append(obs_object.key())
    return obs_names


def get_param_names_for_node(param_node_name, ert):
    ens_config = ert.ensembleConfig()
    node = ens_config.getNode(param_node_name)
    impl_type = node.getImplementationType()
    parameter_names_for_node = []
    if impl_type == ErtImplType.GEN_KW:
        kw_config_model = node.getKeywordModelConfig()
        parameter_names_for_node = kw_config_model.getKeyWords()
    else:
        print(f"Node with type {impl_type} is not yet implemented. Ignore this.")
    return parameter_names_for_node


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
        impl_type = node.getImplementationType()
        var_type = node.getVariableType()
        model_param_node_names.append([key, impl_type, var_type])
    return model_param_node_names


def get_param_nodes_from_gen_kw(ert):
    ens_config = ert.ensembleConfig()
    keylist = ens_config.getKeylistFromImplType(ErtImplType.GEN_KW)
    parameters_for_node = {}
    for key in keylist:
        node = ens_config.getNode(key)
        impl_type = node.getImplementationType()
        if impl_type == ErtImplType.GEN_KW:
            kw_config_model = node.getKeywordModelConfig()
            parameters_for_node[key] = kw_config_model.getKeyWords()
    return parameters_for_node


def get_param_nodes_from_gen_data(ert):
    ens_config = ert.ensembleConfig()
    keylist = ens_config.getKeylistFromVarType(EnkfVarType.PARAMETER)
    parameters_for_node = {}
    for key in keylist:
        node = ens_config.getNode(key)
        impl_type = node.getImplementationType()
        if impl_type == ErtImplType.GEN_DATA:
            parameters_for_node[key] = impl_type
    return parameters_for_node


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

        model_param_node_names = get_param_node_name_list_from_ert(ert)
        if debug_level > 0:
            print(" -- Parameter nodes (name, impl_type, var_type):")
            for item in model_param_node_names:
                name = item[0]
                param_type = item[1]
                var_type = item[2]
                print(f"     {name}  {param_type}  {var_type}")

        parameters_for_node_dict = get_param_nodes_from_gen_kw(ert)
        debug_print(" - Parameter nodes from GEN_KW:")
        for node_name, parameter_list in parameters_for_node_dict.items():
            if debug_level > 0:
                print(f" -- Parameter node: {node_name}")
                for name in parameter_list:
                    print(f"     Parameter name:  {name}")
                print("\n")
        parameters_for_node_from_gen_data = get_param_nodes_from_gen_data(ert)
        for node_name, parameter_type in parameters_for_node_from_gen_data.items():
            if debug_level > 0:
                print(f" -- Parameter node: {node_name}   Type: {parameter_type}")
        print("\n")

        result_node_names = get_result_node_name_list_from_ert(ert)
        if debug_level > 0:
            print(" -- Result nodes:")
            for name in result_node_names:
                print(f"     {name}")
