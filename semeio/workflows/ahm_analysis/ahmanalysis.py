import collections
import itertools
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xtgeo
from ert.analysis import ErtAnalysisError
from ert.shared.plugins.plugin_manager import hook_implementation
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from semeio._exceptions.exceptions import ValidationError
from semeio.communication import SemeioScript

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
        # pylint: disable=method-hidden, too-many-statements, arguments-differ
        # (SemeioScript wraps this run method)

        # pylint: disable=too-many-locals

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
        prior_data = self.facade.load_all_gen_kw_data(
            self.storage.get_ensemble_by_name(prior_name)
        )
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
        field_output = {}
        for group_name, obs_group in combinations.items():
            print("Processing:", group_name)

            #  Use localization to evaluate change of parameters for each observation
            with tempfile.TemporaryDirectory() as update_log_path:
                try:
                    prior_ensemble = self.storage.get_ensemble_by_name(prior_name)
                    target_ensemble = self.storage.create_ensemble(
                        prior_ensemble.experiment_id,
                        name=target_name,
                        ensemble_size=prior_ensemble.ensemble_size,
                        prior_ensemble=prior_ensemble,
                    )
                    _run_ministep(
                        self.facade,
                        prior_ensemble,
                        target_ensemble,
                        obs_group,
                        field_parameters + scalar_parameters,
                        update_log_path,
                    )
                    # Get the active vs total observation info
                    df_update_log = make_update_log_df(update_log_path)
                except ErtAnalysisError:
                    df_update_log = pd.DataFrame(
                        columns=[
                            "obs_key",
                            "obs_mean",
                            "obs_std",
                            "status",
                            "sim_mean",
                            "sim_std",
                        ]
                    )
            # Get the updated scalar parameter distributions
            self.reporter.publish_csv(
                group_name, self.facade.load_all_gen_kw_data(target_ensemble)
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
                    self.facade.load_all_gen_kw_data(target_ensemble),
                )
            )
            field_output[group_name] = _get_field_params(
                self.facade, field_parameters, target_ensemble
            )
        kolmogorov_smirnov_data.set_index("Parameters", inplace=True)

        # Calculate Ks matrix for Fields parameters
        if field_parameters:
            # Get grid characteristics to be able to plot field avg maps
            grid_xyzcenter = load_grid_to_dataframe(self.facade.grid_file)
            all_input_prior = _get_field_params(
                self.facade, field_parameters, prior_ensemble
            )

            for fieldparam in field_parameters:
                scaler = StandardScaler()
                scaler.fit(all_input_prior[fieldparam])
                pca = PCA(0.98).fit(
                    pd.DataFrame(scaler.transform(all_input_prior[fieldparam]))
                )
                pc_fieldprior_df = pd.DataFrame(
                    data=pca.transform(scaler.transform(all_input_prior[fieldparam]))
                )
                all_kolmogorov_smirnov = pd.DataFrame(
                    pc_fieldprior_df.columns.tolist(), columns=["PCFieldParameters"]
                )
                # Get the posterior Field parameters
                map_calc_properties = (
                    grid_xyzcenter[grid_xyzcenter["KZ"] == 1].copy().reset_index()
                )
                for group_name in combinations.keys():
                    map_calc_properties["Mean_D_" + group_name] = calc_mean_delta_grid(
                        field_output[group_name][fieldparam],
                        all_input_prior[fieldparam],
                        grid_xyzcenter,
                    )

                    pc_fieldpost_df = pd.DataFrame(
                        data=pca.transform(
                            scaler.transform(field_output[group_name][fieldparam])
                        )
                    )
                    all_kolmogorov_smirnov[group_name] = all_kolmogorov_smirnov[
                        "PCFieldParameters"
                    ].map(
                        calc_kolmogorov_smirnov(
                            pc_fieldpost_df,
                            pc_fieldprior_df,
                            pc_fieldpost_df,
                        )
                    )
                all_kolmogorov_smirnov.set_index("PCFieldParameters", inplace=True)
                # add the field max Ks to the scalar Ks matrix
                kolmogorov_smirnov_data.loc[
                    "FIELD_" + fieldparam
                ] = all_kolmogorov_smirnov.max()
                self.reporter.publish_csv(
                    "delta_field" + fieldparam, map_calc_properties
                )
        # save/export the Ks matrix, active_obs, misfitval and prior data
        self.reporter.publish_csv("ks", kolmogorov_smirnov_data)
        self.reporter.publish_csv("active_obs_info", active_obs)
        self.reporter.publish_csv("misfit_obs_info", misfitval)
        self.reporter.publish_csv("prior", prior_data)


def _run_ministep(
    facade, prior_storage, target_storage, obs_group, data_parameters, output_path
):
    # pylint: disable=too-many-arguments
    update_step = {
        "name": "MINISTEP",
        "observations": obs_group,
        "parameters": data_parameters,
    }
    facade.update_configuration = [update_step]
    facade.set_log_path(output_path)
    # Perform update analysis

    facade.smoother_update(prior_storage, target_storage, target_storage.name)


def _get_field_params(facade, field_parameters, target_ensemble):
    """
    Because the FIELD parameters are not exposed in the Python API we have to
    export them to file and read them back again. When they are exposed in the API
    this function should be updated.
    """
    field_data = {}
    for field_param in field_parameters:
        dataset = target_ensemble.load_parameters(
            field_param, list(range(facade.get_ensemble_size()))
        )
        field_data[field_param] = dataset.values.reshape(
            facade.get_ensemble_size(), len(dataset.x) * len(dataset.y) * len(dataset.z)
        )
    return field_data


def make_update_log_df(update_log_dir):
    """Read update_log file to get active and inactive observations"""
    list_of_files = [
        os.path.join(update_log_dir, f) for f in os.listdir(update_log_dir)
    ]
    if len(list_of_files) > 1:
        raise OSError("ERROR more than one update_log_file.")
    if len(list_of_files) == 0:
        raise OSError("ERROR empty update_log directory.")
    # read file
    updatelog = pd.read_csv(
        list_of_files[0],
        delim_whitespace=True,
        skiprows=6,
        usecols=[2, 3, 5, 6, 8, 10],
        header=None,
        engine="python",
        skipfooter=1,
    )
    # define header
    updatelog.columns = [
        "obs_key",
        "obs_mean",
        "obs_std",
        "status",
        "sim_mean",
        "sim_std",
    ]

    # ---------------------------------
    # add proper name for obs_key in rows where value is '...'
    updatelog.replace("...", np.nan, inplace=True)
    updatelog.ffill(inplace=True)

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
        missing_obs = [x for x in key_map.keys() if x not in set(subset)]
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
        warnings.warn("WARNING: no MISFIT value for observation " + obs_keys)
    else:
        df_misfit_calc = pd.DataFrame()
        df_misfit_calc["Misfit_key"] = "MISFIT:" + df_update_log[
            df_update_log.status.isin(["Active", "Inactive"])
        ]["obs_key"].astype(str)
        df_misfit_calc = pd.DataFrame.drop_duplicates(df_misfit_calc)
        mean = (
            misfit_df[df_misfit_calc["Misfit_key"].to_list()].sum(axis=1) / total_obs_nr
        )
    return mean.mean()


def load_grid_to_dataframe(grid_path):
    """Get field grid characteristics/coordinates"""
    grid_path = Path(grid_path).with_suffix("")
    try:
        grid = xtgeo.grid_from_file(grid_path, fformat="eclipserun")
        return grid.get_dataframe(activeonly=False)
    except OSError as err:
        raise OSError("A grid with .EGRID format is expected.") from err


def calc_mean_delta_grid(all_input_post, all_input_prior, grid):
    """calculate mean delta of field grid data"""
    delta_post_prior = np.subtract(all_input_prior, all_input_post)
    delta_post_prior = np.absolute(delta_post_prior)
    # Can we avoid changing in place here?
    grid["Mean_delta"] = np.mean(delta_post_prior, axis=0)
    df_mean_delta = grid.groupby(["IX", "JY"])[["Mean_delta"]].mean().reset_index()

    return df_mean_delta["Mean_delta"]


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
            warnings.warn("WARNING: Parameter " + dkey + " defined several times.")
            flatten_arr = np.ravel(prior_data[dkey])
            result = np.all(prior_data[dkey] == flatten_arr[0])
            if not result:
                p_keysf.append(dkey)
        else:
            if not all(x == prior_data[dkey][0] for x in prior_data[dkey]):
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
