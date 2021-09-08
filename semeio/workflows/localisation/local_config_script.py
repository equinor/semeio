from ert_shared.libres_facade import LibresFacade
from ert_shared.plugins.plugin_manager import hook_implementation

# from semeio._docs_utils._json_schema_2_rst import _create_docs

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


DESCRIPTION = """
===================
Localisation setup
===================
LOCALISATION_JOB is used to define which pairs of model parameters and
observations to be active and which pairs to have reduced or 0 correlation.
If no localisation is specified, all model parameters and observations may be
correlated, although correlations can be small. With a finite ensemble of
realisations the estimate of correlations will have sampling uncertainty and
unwanted or unphysical correlations may appear.

By using the localisation job, it is possible to restrict the allowed correlations
or reduce the correlations by a factor between 0 and 1.

Features
----------
The following features are implemented:

 - The user define groups of model parameters and observations.
 - Wildcard notation can be used to specify a selection of model parameter groups
   and observation groups.
 - For scalar parameters coming from the ERT keyword GEN_KW and GEN_PARAM,
   the correlation with observations can be specified to be active or inactive.
 - For field parameters coming from the ERT keyword FIELD and SURFACE,
   it is also possible to specify that the correlation between observations and
   model parameters may vary from location to location. A field parameter
   value corresponding to a grid cell (i,j,k) in location (x,y,z) are reduced by a
   scaling factor varying by distance from a reference point e.g at a location (X,Y,Z),
   usually specified to be close to an observation group.
 - Multiple pairs of groups of model parameters and observations can be specified
   to have active correlations.


Using the localisation setup in ERT
-------------------------------------

To setup localisation:
 - Specify a YAML format configuration file for localisation.
 - Create a workflow file containing the line:
   LOCALISATION_JOB <localisation_config_file>
 - Specify to load the workflow file in the ERT config file using
   LOAD_WORKFLOW keyword in ERT.
 - Specify to automatically run the workflow after initial ensemble is created,
   but before the first update by using the keyword using the
   HOOK_WORKFLOW keyword with the option PRE_FIRST_UPDATE.
"""

EXAMPLES = """
Example configuration
-------------------------

The configuration file is a YAML format file where pairs of groups of observations
and groups of model parameters are specified.

Per default, all correlations between the
observations from the observation group and model parameters from the model
parameter group are active and unmodified. All other combinations of pairs of
observations and model parameters not specified in a correlation group, is inactive
or set to 0. But it is possible to specify many correlation groups. If a pair of
observation and model parameter appear multiple times
(e.g. because they are member of multiple correlation groups),
an error message is raised.

It is also possible to scale down correlations that are specified for 3D and 2D fields.
In the example below, the first correlation group is called ``CORR1`` , a user
defined name, define all observations to have active correlation with all model
parameters starting with ``aps_``. The keyword ``field_scale`` define a scaling of
the correlations between the observations in the group and the model parameters
selected which are of type FIELD. In this example two correlation groups are defined.
The first group( with name ``CORR1`` ) activates correlations with parameters
starting with ``aps_`` while the second correlation group (with name ``CORR2`` )
activates correlations with all parameters except those starting with ``aps_``::

  log_level:3
  correlations:
    - name: CORR1
       obs_group:
          add: ["*"]
       param_group:
          add: ["aps_*"]
       field_scale:
          method: gaussian_decay
          main_range: 1700
          perp_range: 850
          azimuth: 310
       ref_point: [463400, 5932915]

    - name: CORR2
       obs_group:
          add: ["*"]
       param_group:
          add: ["*"]
          remove: ["aps_*"]
       surface_scale:
          method: exponential_decay
          main_range: 800
          perp_range: 350
          azimuth: 120
       ref_point: [463000, 5932850]

Keywords
-----------
:log_level:
      Optional. Define how much information to write to the log file.

:correlations:
      List of specifications of correlation groups.

:name:
      Name of correlation groups.

:obs_group:
      Define  which observations belong to the group.

:param_group:
      Define which model parameters to belong to the group.

:field_scale:
      Optional.
      Define how correlations between field parameters and observations
      in the observation group are modified. For distance based localisation
      typically the correlations are reduced by distance from the observations
      to field parameter value. A reference point is specified in separate keyword
      and should usually be located close to the observations in the observation group
      when using scaling of correlations between field parameters and observations.

:surface_scale:
      Optional.
      Similar to fields, surface parameters are also field parameters, but in 2D.
      Scaling of this is also done in a similar way as for 3D field parameters.

:ref_point:
      Optional but required if  **field_scale**  or **surface_scale** keywords are used.
      The keyword is followed by a list of x and y coordinates for the reference point.

:add:
      Sub keyword for specification of obs_group and param_group. Both **add**
      and **remove** keywords is followed by a list of observations or
      parameter names. Wildcard notation can be specified, and all observations
      or parameters specified in the ERT config file matching the wildcard expansion
      is included.

      The keyword **add** will add new observations or parameters to the list of
      selected observations or parameters while the keyword **remove** will remove
      the specified observations or parameter from the selection.

      Specification of parameters in the list is of the form
      ``node_name:parameter_name`` where node_name is an ERT identifier and
      parameter_name is the name of a parameter belonging to the ERT node.

      For instance if the ``GEN_KW`` ERT keyword is used, the ERT identifier is
      the node name while the parameter names used in the distribution file contains
      names of the parameters for that node.

      For ERT node of type ``GEN_PARAM`` the parameter names are only referred to
      by indices, no names, so in this case the parameter index is specified instead
      such that a parameter in a GEN_PARAM node is referred to
      by ``node_name:index``

:remove:
      For details see the keyword **add:**
:method:
      Define a method for calculating the scaling factor. It depends on whether it
      is a part of the **field_scale** or **surface_scale** keyword.
      Available methods are **gaussian_decay** and **exponential_decay**.
      Parameters defining these functions are defined by the
      keywords **main_range**,  **perp_range**  and **azimuth**



"""


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(LocalisationConfigJob, "LOCALISATION_JOB")
    #    schema = LocalisationConfig.schema(by_alias=False, ref_template="{model}")
    workflow.description = DESCRIPTION
    workflow.examples = EXAMPLES
    #    workflow.description = DESCRIPTION + " " + _create_docs(schema)
    workflow.category = "observations.correlation"
