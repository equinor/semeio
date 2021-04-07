# Localisation configuration test example

import yaml
from  functools import partial

import cwrap
from res.enkf import ErtScript, EnkfObs
from ecl.grid import EclRegion, EclGrid
from ecl.eclfile import EclKW, Ecl3DKW
from ecl.ecl_type import EclDataType 
from ecl.util.util import IntVector



#Global variables
debug_level = 1

def read_localisation_config(specification_file_name, debug_level=0):
    if debug_level >= 1:
        print('\n')
        print(f'Localisation config file: {specification_file_name}')
    with open(specification_file_name, 'r') as yml_file:
        localisation_yml = yaml.safe_load(yml_file)
    all_keywords = localisation_yml['localisation']
    if all_keywords is None:
        raise IOError(f'None is found when reading file: {specification_file_name}')
    return all_keywords

def get_defined_scalar_parameters(defined_scalar_parameters, 
                                  ens_config, 
                                  debug_level=0):
    node_with_scalar_params = {}
    for node_name, param_definition in defined_scalar_parameters.items():
        # Check if node name exist in ert configuration
        if node_name not in ens_config:
            raise ValueError(f'Node name {node_name} is not defined in ERT config file\n')
        # Get list of defined parameter names for this node
        file_name = param_definition['parameter_distribution_file']
        parameter_names = []
        if debug_level > 0:
            print(f' -- Define model parameter node: {node_name}')
            print(f' -- Read file: {file_name}')
        with open(file_name, "r") as file:
            lines = file.readlines()
        for line in lines:
            words = line.split()
            parameter_names.append(words[0])
        node_with_scalar_params[node_name] = parameter_names
        if debug_level > 0:
            print(f' -- Parameter names for node {node_name}:')
            for name in parameter_names:
                print(f'      {name}')
    return node_with_scalar_params

class LocalisationConfigJob(ErtScript):
    def run(self):
        #Initialisation
        ert = self.ert()
        local_config = ert.getLocalConfig()
        local_config.clear()
        updatestep = local_config.getUpdatestep() 
        ens_config = ert.ensembleConfig()
        keylist = ens_config.alloc_keylist()
        if debug_level > 0:
            print(f'\n -- Node names for model parameters found in ERT configuration:')
            for node_name in keylist:
                print(f'      {node_name}')

        ert_obs = ert.getObservations()
        obs_data_from_ert_config = ert_obs.getAllActiveLocalObsdata()
        ert_obs_node_list = []
        for obs_node in obs_data_from_ert_config:
            node_name = obs_node.key()
            ert_obs_node_list.append(node_name)

        if debug_level > 0:
            print(f'\n -- Node names for observations found in ERT configuration:')
            for node_name in  ert_obs_node_list:
                print(f'      {node_name}')



        # Read yml file with specifications
        spec_file_name = 'localisation_config_file_scalar.yml'
        all_kw= read_localisation_config( spec_file_name, debug_level=debug_level)

        # Path to configuration setup
        config_path = all_kw['config_path']

        # Get model parameter groups and associated model parameter nodes
        model_param_groups_spec = all_kw['model_param_groups']

        # Get observation groups and associated observation nodes
        obs_groups_spec = all_kw['obs_groups']

        # Get ministep specifications
        mini_steps_spec = all_kw['ministeps']

        #Get definition of scalar parameter names for each parameter node that is defined

        try:
            defined_scalar_parameters = all_kw['defined_scalar_parameters']
        except:
            defined_scalar_parameters = None
        node_with_scalar_params = {}
        if defined_scalar_parameters is not None:
            node_with_scalar_params =  get_defined_scalar_parameters(defined_scalar_parameters, 
                                                                     ens_config, 
                                                                     debug_level=debug_level)

        # Loop over all mini step specifications and setup the ministeps
        model_param_group_used = {}
        obs_group_used = {}
        for mini_step_name, mini_step_specs in mini_steps_spec.items():
            model_group_name = mini_step_specs['model_group']
            obs_group_name = mini_step_specs['obs_group'] 
            if debug_level > 0:
                print(f' -- Get specification of ministep: {mini_step_name}')
                print(f' -- Use model group:  {model_group_name}  and obs group: {obs_group_name}')

            # Define model_group
            if model_group_name not in model_param_group_used.keys():
                # Created, create new group
                model_param_group = local_config.createDataset(model_group_name) 
                model_param_group_used[model_group_name] = model_param_group
                model_param_group_spec = model_param_groups_spec[model_group_name]
                group_type = model_param_group_spec['type']
                if group_type == 'Scalar':
                    node_name_list = model_param_group_spec['nodes']
                    # Expecting to find nodes that are defined
                    node_name_used =[]
                    for node_name, param_list in node_name_list.items():
                        if node_name in node_with_scalar_params.keys():
                            if debug_level > 0:
                                print(f' -- Node name: {node_name}')
                            if node_name not in node_name_used:
                                node_name_used.append(node_name)
                            else:
                                raise ValueError(f' Node name: {node_name} is specified multiple times for model group name: {model_group_name}')
                            model_param_group.addNode(node_name)
                            params = model_param_group.getActiveList(node_name)
                            parameter_names = node_with_scalar_params[node_name]

                            # Go through the list of specified parameter names for this node, 
                            # check that it is defined and if it is active
                            for pname, isActive in param_list.items():
                                index = -1
                                for n in range(len(parameter_names)):
                                    if pname == parameter_names[n]:
                                        index = n
                                        break
                                if index == -1:
                                    raise ValueError(f"Parameter with name: {pname} is not defined for node: {node_name} in keyword 'defined_scalar_parameters' \n"
                                                     f"Check specification of model parameter group {model_group_name} or node definition in keyword  'defined_scalar_parameters' ")
                                if isActive:
                                    if debug_level > 0:
                                        print(f' -- Set parameter: {pname} in node {node_name} in model group {model_group_name} active')
                                    params.addActiveIndex(index)
                        else:
                            raise ValueError(f"The node with name: {node_name}  found under model param group {model_group_name}\n"
                                             f"is  not defined in keyword: 'defined_scalar_parameters' ")
                else:
                    raise ValueError('Nodes that are not scalars are not yet implemented')
            else:
                model_param_group = model_param_group_used[model_group_name]

            # Define obs group
            if obs_group_name not in obs_group_used.keys():
                # Create new obs group
                obs_group = local_config.createObsdata(obs_group_name)
                obs_group_used[obs_group_name] = obs_group
                obs_group_spec = obs_groups_spec[obs_group_name]
                obs_nodes_list = obs_group_spec['nodes']
                for obs_node_name in obs_nodes_list:
                    if debug_level > 0:
                        print(f' -- Add node: {obs_node_name} in  group: {obs_group_name}')
                    if obs_node_name in ert_obs_node_list:
                        obs_group.addNode(obs_node_name)
                    else:
                        raise ValueError(f'Obs node name {obs_node_name} is not defined in ERT config file\n')
            else:
                obs_group = obs_group_used[obs_group_name]

            # Define mini step object
            ministep = local_config.createMinistep(mini_step_name)
            if debug_level > 0:
                print(f' -- Attach {model_group_name} to {mini_step_name}')
            ministep.attachDataset(model_param_group)
            if debug_level > 0:
                print(f' -- Attach {obs_group_name} to {mini_step_name}')
            ministep.attachObsset(obs_group) 

            if debug_level > 0:
                print(f' -- Add {mini_step_name} to update step\n')
            updatestep.attachMinistep(ministep)

