import collections
import itertools
import logging
import tempfile
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from ert import hook_implementation
from ert.analysis import ErtAnalysisError, SmootherSnapshot
from ert.storage import open_storage
from scipy.stats import ks_2samp

from semeio._exceptions.exceptions import ValidationError
from semeio.communication import SemeioScript

logger = logging.getLogger(__name__)

DESCRIPTION = """
AHM_ANALYSIS will calculate the degree of update (using Kolmogorov Smirnov test)
of the different parameters with regards to the different observations/groups of
observations during the history matching process in ERT.

Results will be exported in csv files which can be used in Webviz to perform
analysis of the AHM through the AhmAnalysis plugin. CSV files will be stored
under a folder named :code:`ENSPATH/../reports/prior_name/AhmAnalysisJob/`.
prior_name will be the name of the case selected in current case on the ERT GUI
(often :code:`default`)or given as optional input to the workflow. Note that
depending on the setup of ENSPATH, files may be overwritten when the workflow
is run for different ert cases. Files generated are:

 - One csv file per observation group with posterior distribution of the
   different parameters for each observation group included in the update.
 - One csv file named :code:`prior` with prior distribution of the different
   parameters.
 - One csv file named :code:`ks` with the results of the Kolmogorov Smirnov test values
   for each parameter for each observation group.
 - One csv file named :code:`misfit_obs_info` with average misfit for each observation
   group.
 - One csv file named :code:`active_obs_info` with indication of number of active
   observation used by ERT in History matching vs total number of observations
   within each group of observations.


Three optional inputs can be given:

 - :code:`target_name`: a name for the target case used for update (ahm)
   calculation to avoid overwriting an existing case (by default the target case
   will be named :code:`analysis_case`).
 - :code:`prior_name`: a name for the prior case to be used (by default the case
   selected as current case name in the ert gui is taken)
 - :code:`group_by`: default is :code:`data_key` to group by observations groups
   as shown on the ERT GUI Observations. Can be changed to :code:`obs_key` to
   perform calculation on individual observations (note: only works for
   :code:`SUMMARY_OBSERVATION` !Warning! this will increase time of calculation
   if a lot of :code:`SUMMARY_OBSERVATION`'s.
"""

EXAMPLES = """
Example:
--------
Add a file named e.g. :code:`AHM_ANALYSIS` with the contents:

::

          AHM_ANALYSIS

then all optional inputs have default values
(:code:`target_name="analysis_case"`, prior_name is from current case selected
in the ERT GUI and :code:`group_by="data_key"` (list of observations as listed
in the ERT GUI Configuration summary part))

Add to your ERT config to have the workflow available for execution through Run
Workflow button on ERT GUI

::

          --Perform the AHM analysis
          LOAD_WORKFLOW AHM_ANALYSIS


To use the optional inputs, in the file named
:code:`ert/bin/workflows/AHM_ANALYSIS` set the inputs to: :code:`AHM_ANALYSIS
<ANALYSIS_CASE_NAME> <PRIOR_CASE_NAME> obs_key`

where :code:`<ANALYSIS_CASE_NAME>` and :code:`<PRIOR_CASE_NAME>` are defined in
the ERT config and add to your ERT config to have the workflow available for
execution through Run Workflow button on ERT GUI

::

          --Perform the AHM analysis
          DEFINE <ANALYSIS_CASE_NAME> sens_ahm_analysis
          DEFINE <PRIOR_CASE_NAME> default
          LOAD_WORKFLOW AHM_ANALYSIS

"""


class AhmAnalysisJob(SemeioScript):
    """Define ERT workflow to evaluate change of parameters for eac
    observation during history matching
    """

    def run(
        self,
        target_name="analysis_case",
        prior_name=None,
        group_by="data_key",
        output_dir=None,
    ):
        # (SemeioScript wraps this run method)

        """Perform analysis of parameters change per obs group
        prior to posterior of ahm"""
        if output_dir is not None:
            self._reports_dir = output_dir

        obs_keys = list(self.facade.get_observations().obs_vectors.keys())
        key_map = _group_observations(self.facade, obs_keys, group_by)

        prior_name, target_name = check_names(
            self.ensemble.name,
            prior_name,
            target_name,
        )
        # Get the prior scalar parameter distributions
        ensemble = self.storage.get_ensemble_by_name(prior_name)
        prior_data = ensemble.load_all_gen_kw_data()
        try:
            raise_if_empty(
                dataframes=[
                    prior_data,
                    self.facade.load_all_misfit_data(
                        self.storage.get_ensemble_by_name(prior_name)
                    ),
                ],
                messages=[
                    "Empty prior ensemble",
                    "Empty parameters set for History Matching",
                ],
            )
        except KeyError as err:
            raise ValidationError(f"Empty prior ensemble: {err}") from err

        # create dataframe with observations vectors (1 by 1 obs and also all_obs)
        combinations = make_obs_groups(key_map)

        field_parameters = sorted(self.facade.get_field_parameters())
        if field_parameters:
            logger.warning(
                f"AHM_ANALYSIS will only evaluate scalar parameters, skipping: {field_parameters}"
            )

        scalar_parameters = sorted(self.facade.get_gen_kw())
        # identify the set of actual parameters that was updated for now just go
        # through scalar parameters but in future if easier access to field parameter
        # updates should also include field parameters
        dkeysf = get_updated_parameters(prior_data, scalar_parameters)
        # setup dataframe for calculated data
        kolmogorov_smirnov_data, active_obs, misfitval = (
            pd.DataFrame(sorted(dkeysf), columns=["Parameters"]),
            pd.DataFrame(),
            pd.DataFrame(index=["misfit"]),
        )
        # loop over keys and calculate the KS matrix,
        # conditioning one parameter at the time.
        updated_combinations = deepcopy(combinations)
        for group_name, obs_group in combinations.items():
            print("Processing:", group_name)

            #  Use localization to evaluate change of parameters for each observation
            # The order of the context managers is important, as we want to create a new
            # storage in a temporary directory.
            with tempfile.TemporaryDirectory(), open_storage(
                "tmp_storage", "w"
            ) as storage:
                try:
                    prior_ensemble = self.storage.get_ensemble_by_name(prior_name)
                    prev_experiment = prior_ensemble.experiment
                    experiment = storage.create_experiment(
                        parameters=prev_experiment.parameter_configuration.values(),
                        observations=prev_experiment.observations,
                        responses=prev_experiment.response_configuration.values(),
                    )
                    target_ensemble = storage.create_ensemble(
                        experiment,
                        name=target_name,
                        ensemble_size=prior_ensemble.ensemble_size,
                    )
                    update_log = _run_ministep(
                        self.facade,
                        prior_ensemble,
                        target_ensemble,
                        obs_group,
                        field_parameters + scalar_parameters,
                    )
                    # Get the active vs total observation info
                    df_update_log = make_update_log_df(update_log)
                except ErtAnalysisError:
                    logger.error(f"Analysis failed for: {obs_group}")
                    del updated_combinations[group_name]
                    continue
                # Get the updated scalar parameter distributions
                self.reporter.publish_csv(
                    group_name, target_ensemble.load_all_gen_kw_data()
                )

                active_obs.at["ratio", group_name] = (
                    str(count_active_observations(df_update_log))
                    + " active/"
                    + str(len(df_update_log.index))
                )
                # Get misfit values
                misfitval[group_name] = [
                    calc_observationsgroup_misfit(
                        group_name,
                        df_update_log,
                        self.facade.load_all_misfit_data(prior_ensemble),
                    )
                ]
                # Calculate Ks matrix for scalar parameters
                kolmogorov_smirnov_data[group_name] = kolmogorov_smirnov_data[
                    "Parameters"
                ].map(
                    calc_kolmogorov_smirnov(
                        dkeysf,
                        prior_data,
                        target_ensemble.load_all_gen_kw_data(),
                    )
                )
        kolmogorov_smirnov_data.set_index("Parameters", inplace=True)

        # save/export the Ks matrix, active_obs, misfitval and prior data
        self.reporter.publish_csv("ks", kolmogorov_smirnov_data)
        self.reporter.publish_csv("active_obs_info", active_obs)
        self.reporter.publish_csv("misfit_obs_info", misfitval)
        self.reporter.publish_csv("prior", prior_data)


def _run_ministep(facade, prior_storage, target_storage, obs_group, data_parameters):
    rng = np.random.default_rng(seed=facade.config.random_seed)
    return facade.smoother_update(
        prior_storage,
        target_storage,
        target_storage.name,
        obs_group,
        data_parameters,
        rng=rng,
    )


def make_update_log_df(update_log: SmootherSnapshot) -> pd.DataFrame:
    """Read update_log file to get active and inactive observations"""
    obs_key = []
    obs_mean = []
    obs_std = []
    sim_mean = []
    sim_std = []
    status = []

    # Loop through each update_step_snapshot once, collecting all necessary information
    for step in update_log.update_step_snapshots:
        obs_key.append(step.obs_name)
        obs_mean.append(step.obs_val)
        obs_std.append(step.obs_std)
        sim_mean.append(step.response_mean)
        sim_std.append(step.response_std)
        status.append(
            "Active"
            if step.response_mean_mask and step.response_std_mask
            else "Inactive"
        )

    updatelog = pd.DataFrame(
        {
            "obs_key": obs_key,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "status": status,
            "sim_mean": sim_mean,
            "sim_std": sim_std,
        }
    )

    return updatelog


def make_obs_groups(key_map):
    """Create a mapping of observation groups, the names will be:
    data_key -> [obs_keys] and All_obs-{missing_obs} -> [obs_keys]
    and All_obs -> [all_obs_keys]
    """
    combinations = key_map.copy()
    if len(combinations) == 1:
        return combinations

    combinations["All_obs"] = list(itertools.chain.from_iterable(key_map.values()))

    if len(combinations) == 3:
        return combinations

    for subset in itertools.combinations(key_map.keys(), len(key_map.keys()) - 1):
        obs_group = list(
            itertools.chain.from_iterable([key_map[key] for key in subset])
        )
        missing_obs = [x for x in key_map if x not in set(subset)]
        assert len(missing_obs) == 1
        name = f"All_obs-{missing_obs[0]}"
        combinations[name.replace(":", "_")] = obs_group

    return combinations


def count_active_observations(df_update_log):
    """To get the active observation info."""
    df_active = df_update_log[(df_update_log["status"] == "Active")]
    return len(df_active)


def calc_observationsgroup_misfit(obs_keys, df_update_log, misfit_df):
    """To get the misfit for total observations (active/inactive)."""

    total_obs_nr = len(df_update_log[df_update_log.status.isin(["Active", "Inactive"])])
    if total_obs_nr == 0:
        mean = pd.DataFrame({0: [-999]}).loc[0]
        warnings.warn(
            "WARNING: no MISFIT value for observation " + obs_keys, stacklevel=1
        )
    else:
        df_misfit_calc = pd.DataFrame()
        df_misfit_calc["Misfit_key"] = "MISFIT:" + df_update_log[
            df_update_log.status.isin(["Active", "Inactive"])
        ]["obs_key"].astype(str)
        df_misfit_calc = pd.DataFrame.drop_duplicates(df_misfit_calc)
        mean = (
            misfit_df[df_misfit_calc.loc[:, "Misfit_key"].to_list()].sum(axis=1)
            / total_obs_nr
        )
    return mean.mean()


def _filter_on_prefix(list_of_strings, prefixes):
    """returns the set of strings that has a match for any of the given prefixes"""
    return {
        string
        for string in list_of_strings
        if any(string.startswith(prefix) for prefix in prefixes)
    }


def get_updated_parameters(prior_data, parameters):
    """make list of updated parameters
    (excluding duplicate transformed parameters)
    """
    parameter_keys = _filter_on_prefix(
        list_of_strings=prior_data.keys(), prefixes=parameters
    )
    # remove parameters with constant prior distribution
    p_keysf = []
    for dkey in parameter_keys:
        if prior_data[dkey].ndim > 1:
            warnings.warn(
                "WARNING: Parameter " + dkey + " defined several times.", stacklevel=1
            )
            flatten_arr = np.ravel(prior_data[dkey])
            result = np.all(prior_data[dkey] == flatten_arr[0])
            if not result:
                p_keysf.append(dkey)
        elif not all(x == prior_data[dkey][0] for x in prior_data[dkey]):
            p_keysf.append(dkey)
    return p_keysf


def calc_kolmogorov_smirnov(columns, prior_data, target_data):
    """Calculate kolmogorov_smirnov matrix"""
    ks_param = {}
    for dkey in sorted(columns):
        ks_val = ks_2samp(prior_data[dkey], target_data[dkey])
        ks_param[dkey] = ks_val[0]
    return ks_param


def check_names(ert_currentname, prior_name, target_name):
    """Check input names given"""
    if prior_name is None:
        prior_name = ert_currentname
    if target_name == "<ANALYSIS_CASE_NAME>":
        target_name = "analysis_case"
    return prior_name, target_name


def raise_if_empty(dataframes, messages):
    """Check input ensemble prior is not empty
    and if ensemble contains parameters for hm"""
    for dframe in dataframes:
        if dframe.empty:
            raise ValidationError(f"{messages}")


def _group_observations(facade, obs_keys, group_by):
    key_map = collections.defaultdict(list)
    for obs_key in obs_keys:
        if group_by == "data_key":
            key = facade.get_data_key_for_obs_key(obs_key)
        else:
            key = obs_key
        key_map[key.replace(":", "_")].append(obs_key.replace(":", "_"))
    return key_map


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(AhmAnalysisJob, "AHM_ANALYSIS")
    workflow.description = DESCRIPTION
    workflow.examples = EXAMPLES
    workflow.category = "Analysis"
