import contextlib
import copy
import itertools
import logging
import os
import tempfile
import warnings
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import ert
import numpy as np
import pandas as pd
import polars as pl
from ert import ErtScript, LibresFacade
from ert.analysis import (
    ErtAnalysisError,
    ObservationStatus,
    SmootherSnapshot,
    smoother_update,
)
from ert.config import ESSettings, Field, GenKwConfig, ObservationSettings
from ert.storage import Ensemble, Storage, open_storage
from scipy.stats import ks_2samp

from semeio._exceptions.exceptions import ValidationError

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


class AhmAnalysisJob(ErtScript):
    """Define ERT workflow to evaluate change of parameters for eac
    observation during history matching
    """

    def run(
        self,
        workflow_args: list[str],
        storage: Storage,
        es_settings: ESSettings,
        observation_settings: ObservationSettings,
        random_seed: int,
        reports_dir: str,
        ensemble: Ensemble | None,
    ) -> Any:
        """Perform analysis of parameters change per obs group
        prior to posterior of ahm"""

        target_name = workflow_args[0] if len(workflow_args) > 0 else "analysis_case"
        prior_name = workflow_args[1] if len(workflow_args) > 1 else None
        group_by = workflow_args[2] if len(workflow_args) > 2 else "data_key"

        prior_ensemble = None
        if ensemble is None:
            with contextlib.suppress(KeyError):
                for _experiment in storage.experiments:
                    ensemble = _experiment.get_ensemble_by_name(prior_name)

        prior_ensemble = ensemble

        if prior_ensemble is None:
            all_ensemble_names = [e.name for e in storage.ensembles]
            raise ValueError(
                f"Prior ensemble with name: {prior_name} "
                f"not found in any experiments. "
                f"Available ensemble names: {', '.join(all_ensemble_names)}"
            )

        assert prior_ensemble is not None

        prior_experiment = prior_ensemble.experiment
        observations_and_responses_mapping = (
            pl.concat(
                df["observation_key", "response_key"]
                for df in prior_experiment.observations.values()
            )
            if len(prior_experiment.observations) > 0
            else pl.DataFrame({"observation_key": [], "response_key": []})
        )

        def _replace(s: str) -> str:
            return s.replace(":", "_")

        if group_by != "data_key":
            key_map = {
                _replace(k): [k]
                for k in observations_and_responses_mapping["observation_key"].unique()
            }
        else:
            key_2_obs_key_df = observations_and_responses_mapping.group_by(
                "response_key"
            ).agg(pl.col("observation_key").unique())

            key_map = {
                _replace(l["response_key"]): sorted(map(_replace, l["observation_key"]))
                for l in key_2_obs_key_df.sort(by="response_key").to_dicts()
            }

        if target_name == "<ANALYSIS_CASE_NAME>":
            target_name = "analysis_case"

        prior_data = prior_ensemble.load_all_gen_kw_data()
        try:
            raise_if_empty(
                dataframes=[
                    prior_data,
                    LibresFacade.load_all_misfit_data(prior_ensemble),
                ],
                messages=[
                    "Empty prior ensemble",
                    "Empty parameters set for History Matching",
                ],
            )
        except KeyError as err:
            raise ValidationError(f"Empty prior ensemble: {err}") from err

        ahmanalysis_reports_dir = Path(reports_dir) / "AhmAnalysisJob"
        os.makedirs(ahmanalysis_reports_dir, exist_ok=True)

        # create dataframe with observations vectors (1 by 1 obs and also all_obs)
        combinations = make_obs_groups(key_map)

        field_parameters = [
            p.name
            for p in prior_experiment.parameter_configuration.values()
            if isinstance(p, Field)
        ]
        gen_kws = [
            p.name
            for p in prior_experiment.parameter_configuration.values()
            if isinstance(p, GenKwConfig)
        ]
        if field_parameters:
            logger.warning(
                f"AHM_ANALYSIS will only evaluate scalar parameters, skipping: {field_parameters}"
            )

        scalar_parameters = sorted(gen_kws)
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
            with (
                tempfile.TemporaryDirectory(),
                open_storage("tmp_storage", "w") as tmp_storage,
            ):
                try:
                    target_experiment = tmp_storage.create_experiment(
                        parameters=prior_experiment.parameter_configuration.values(),
                        observations=prior_experiment.observations,
                        responses=prior_experiment.response_configuration.values(),
                    )
                    target_ensemble = tmp_storage.create_ensemble(
                        target_experiment,
                        name=target_name,
                        ensemble_size=prior_ensemble.ensemble_size,
                    )
                    update_log = _run_ministep(
                        prior_storage=prior_ensemble,
                        target_storage=target_ensemble,
                        obs_group=obs_group,
                        data_parameters=field_parameters + scalar_parameters,
                        observation_settings=observation_settings,
                        es_settings=es_settings,
                        random_seed=random_seed,
                    )
                    # Get the active vs total observation info
                    df_update_log = make_update_log_df(update_log)
                except ErtAnalysisError:
                    logger.error(f"Analysis failed for: {obs_group}")
                    del updated_combinations[group_name]
                    continue
                # Get the updated scalar parameter distributions
                target_ensemble.load_all_gen_kw_data().to_csv(
                    ahmanalysis_reports_dir / f"{group_name}.csv"
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
                        LibresFacade.load_all_misfit_data(prior_ensemble),
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

        kolmogorov_smirnov_data.to_csv(ahmanalysis_reports_dir / "ks.csv")
        active_obs.to_csv(ahmanalysis_reports_dir / "active_obs_info.csv")
        misfitval.to_csv(ahmanalysis_reports_dir / "misfit_obs_info.csv")
        prior_data.to_csv(ahmanalysis_reports_dir / "prior.csv")


def make_update_log_df(update_log: SmootherSnapshot) -> pd.DataFrame:
    """Read update_log file to get active and inactive observations"""
    return (
        update_log.observations_and_responses.select(
            "observation_key",
            "observations",
            "std",
            "status",
            "response_mean",
            "response_std",
        )
        .rename(
            {
                "observation_key": "obs_key",
                "observations": "obs_mean",  # not really the mean, just the value, but ok
                "std": "obs_std",
                "response_mean": "sim_mean",
                "response_std": "sim_std",
            }
        )
        .with_columns(
            pl.when(pl.col("status") == ObservationStatus.ACTIVE)
            .then(pl.lit("Active"))
            .otherwise(pl.lit("Inactive"))
            .alias("status")
        )
        .to_pandas()
    )


def _run_ministep(
    prior_storage: Ensemble,
    target_storage: Ensemble,
    obs_group: Iterable[str],
    data_parameters: Iterable[str],
    observation_settings: ObservationSettings,
    es_settings: ESSettings,
    random_seed: int,
) -> SmootherSnapshot:
    rng = np.random.default_rng(random_seed)

    return smoother_update(
        prior_storage=prior_storage,
        posterior_storage=target_storage,
        observations=obs_group,
        parameters=data_parameters,
        update_settings=copy.deepcopy(observation_settings),
        es_settings=es_settings,
        rng=rng,
    )


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


def raise_if_empty(dataframes, messages):
    """Check input ensemble prior is not empty
    and if ensemble contains parameters for hm"""
    for dframe in dataframes:
        if dframe.empty:
            raise ValidationError(f"{messages}")


@ert.plugin(name="semeio")
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(AhmAnalysisJob, "AHM_ANALYSIS")
    workflow.description = DESCRIPTION
    workflow.examples = EXAMPLES
    workflow.category = "Analysis"
