import tempfile
import glob
import os
import itertools
import numpy as np
import pandas as pd

from scipy.stats.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from semeio.communication import SemeioScript
from res.enkf import (
    ErtImplType,
    EnkfNode,
)
from res.enkf import (
    ESUpdate,
    ErtRunContext,
)
from res.enkf.export import (
    GenKwCollector,
    MisfitCollector,
)

from ert_shared.libres_facade import LibresFacade
import collections


DESCRIPTION = """
Purpose is to calculate (ERT workflow)
the degree of update (KS matrix)
of the different parameters with regards to the different
observations. Create csv files which can be used in
Webviz/Jupyternotebook/dash to perform analysis of the AHM.
requires an ert config file as input
optionally a name can be set for the target case used for
update (ahm) calculation
"""


class AhmAnalysisJob(SemeioScript):  # pylint: disable=too-few-public-methods
    """ Define ERT workflow to evaluate change of parameters for each observation"""

    def run(self, target_name="analysis_case", prior_name=None):
        """Perform analysis of parameters change per obs group
        prior to posterior of ahm"""
        ert = self.ert()
        facade = LibresFacade(self.ert())

        obs_keys = [
            facade.get_observation_key(nr)
            for nr, _ in enumerate(facade.get_observations())
        ]

        key_map = collections.defaultdict(list)
        for group_name in obs_keys:
            data_key = facade.get_data_key_for_obs_key(group_name)
            key_map[data_key].append(group_name)

        prior_name, target_name = check_names(
            facade.get_current_case_name(),
            prior_name,
            target_name,
        )
        # Get the prior scalar parameter distributions
        prior_data = check_inputs(
            MisfitCollector.loadAllMisfitData(ert, prior_name),
            GenKwCollector.loadAllGenKwData(ert, prior_name),
        )

        # create dataframe with observations vectors (1 by 1 obs and also all_obs)
        combinations = make_obs_groups(key_map)
        # setup dataframe for calculated data
        ks_data, active_obs, misfitval = initialize_emptydf()

        field_parameters = sorted(
            ert.ensembleConfig().getKeylistFromImplType(ErtImplType.FIELD)
        )
        scalar_parameters = sorted(
            ert.ensembleConfig().getKeylistFromImplType(ErtImplType.GEN_KW)
        )
        # identify the set of actual parameters that was updated for now just go
        # through scalar parameters but in future if easier access to field parameter
        # updates should also include field parameters
        dkeysf = get_updated_parameters(prior_data, scalar_parameters)

        # loop over keys and calculate the KS metric,
        # conditioning one parameter at the time.
        field_output = {}
        for group_name, obs_group in combinations.items():
            print("Processing:", group_name)

            #  Use localization to evaluate change of parameters for each observation
            update_log_path = create_path(
                ert.getModelConfig().getRunpathAsString(), target_name
            ).replace("scalar_", "update_log")

            _run_ministep(
                ert,
                facade,
                obs_group,
                field_parameters + scalar_parameters,
                prior_name,
                target_name,
                update_log_path,
            )
            # Get the updated scalar parameter distributions
            self.reporter.publish_csv(
                "gen_kw", GenKwCollector.loadAllGenKwData(ert, target_name)
            )

            # Get the inactive vs total observation info
            df_update_log = make_update_log_df(update_log_path, group_name)

            active_obs = list_active_observations(
                group_name,
                active_obs,
                df_update_log,
            )
            # Get misfit values
            misfitval.at["misfit", group_name] = list_observations_misfit(
                obs_group,
                df_update_log,
                MisfitCollector.loadAllMisfitData(ert, prior_name),
            )

            # Calculate Ks matrix for scalar parameters
            ks_data = calc_ks(
                ks_data,
                dkeysf,
                prior_data,
                GenKwCollector.loadAllGenKwData(ert, target_name),
                group_name,
            )
            field_output[group_name] = _get_field_params(
                ert, facade.get_ensemble_size(), field_parameters, target_name
            )

        # Calculate Ks matrix for Fields parameters
        if field_parameters:
            # Get grid characteristics to be able to plot field avg maps
            mygrid_ok = get_field_grid_char(ert.eclConfig().get_gridfile())
            all_input_prior = _get_field_params(
                ert, facade.get_ensemble_size(), field_parameters, prior_name
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
                all_ks = pd.DataFrame()
                # Get the posterior Field parameters
                mygrid_ok_short = mygrid_ok[mygrid_ok["KZ"] == 1].copy().reset_index()
                for group_name in combinations.keys():

                    mygrid_ok_short = calc_delta_grid(
                        field_output[group_name][fieldparam],
                        all_input_prior[fieldparam],
                        mygrid_ok,
                        group_name,
                        mygrid_ok_short,
                    )
                    pc_fieldpost_df = pd.DataFrame(
                        data=pca.transform(
                            scaler.transform(field_output[group_name][fieldparam])
                        )
                    )
                    all_ks = calc_ks(
                        all_ks,
                        pc_fieldpost_df,
                        pc_fieldprior_df,
                        pc_fieldpost_df,
                        group_name,
                    )
                # add the field max Ks to the scalar Ks matrix
                ks_data.loc["FIELD_" + fieldparam] = all_ks.max()
                self.reporter.publish_csv("delta_field" + fieldparam, mygrid_ok_short)
        # save/export the Ks matrix, active_obs, misfitval and prior data
        self.reporter.publish_csv("ks", ks_data)
        self.reporter.publish_csv("active_obs_info", active_obs)
        self.reporter.publish_csv("misfit_obs_info", misfitval)
        self.reporter.publish_csv("prior", prior_data)


def _run_ministep(
    ert, facade, obs_group, data_parameters, prior_name, target_name, output_path
):
    # Reset internal local config structure, in order to make your own
    ert.getLocalConfig().clear()

    # A ministep is used to link betwen data and observations.
    # Make more ministeps to condition different groups together
    ministep = ert.getLocalConfig().createMinistep("MINISTEP")
    # Add all dataset to localize
    data_all = ert.getLocalConfig().createDataset("DATASET")
    for data in data_parameters:
        data_all.addNode(data)
    # Add all obs to be used in this updating scheme
    obsdata = ert.getLocalConfig().createObsdata("OBS")
    for obs in obs_group:
        obsdata.addNode(obs)
    # Attach the created dataset and obsset to the ministep
    ministep.attachDataset(data_all)
    ministep.attachObsset(obsdata)
    # Then attach the ministep to the update step
    facade.get_update_step().attachMinistep(ministep)

    # Perform update analysis
    ert.analysisConfig().set_log_path(output_path)
    run_context = ErtRunContext.ensemble_smoother_update(
        ert.getEnkfFsManager().getFileSystem(prior_name),
        ert.getEnkfFsManager().getFileSystem(target_name),
    )
    ESUpdate(ert).smootherUpdate(run_context)


def _get_field_params(ert, ensemble_size, field_parameters, target_name):
    """
    Because the FIELD parameters are not exposed in the Python API we have to
    export them to file and read them back again. When they are exposed in the API
    this function should be updated.
    """
    field_data = {}
    file_system = ert.getEnkfFsManager().getFileSystem(target_name)
    with tempfile.TemporaryDirectory() as fout:
        for field_param in field_parameters:
            config_node = ert.ensembleConfig()[field_param]
            ext = config_node.get_enkf_outfile().rsplit(".")[-1]
            file_path = os.path.join(fout, "%d_" + field_param + "." + ext)
            _export_field_param(config_node, file_system, ensemble_size, file_path)
            fnames = glob.glob(os.path.join(fout, "*" + field_param + "*"))
            field_data[field_param] = _import_field_param(
                ert.eclConfig().get_gridfile(), field_param, fnames
            )
    return field_data


def _import_field_param(input_grid, param_name, files):
    mygrid_char = Grid(input_grid.rsplit(".", 1)[0], fformat="eclipserun")
    all_input = []
    for file_path in files:
        proproff = GridProperty(file_path, name=param_name, grid=mygrid_char)
        array_nb = proproff.get_npvalues1d(activeonly=False, fill_value=0, order="C")
        all_input.append(array_nb)
    return all_input


def _export_field_param(config_node, file_system, ensemble_size, output_path):
    # Get/export the updated Field parameters
    EnkfNode.exportMany(
        config_node,
        output_path,
        file_system,
        np.arange(0, ensemble_size),
    )


def initialize_emptydf():
    ks_data = pd.DataFrame()
    active_obs = pd.DataFrame()
    misfitval = pd.DataFrame()
    return ks_data, active_obs, misfitval


def make_update_log_df(update_log_file, key):
    """ Read update_log file to get active and inactive observations """
    list_of_files = [
        os.path.join(update_log_file, f) for f in os.listdir(update_log_file)
    ]
    if len(list_of_files) > 1:
        raise OSError("Warning more than one update_log_file.")
    # read file
    updatelog = pd.read_csv(
        list_of_files[0],
        delim_whitespace=True,
        skiprows=6,
        usecols=[0, 2, 3, 5, 6, 8, 10],
        header=None,
    )
    # define header
    updatelog.columns = [
        "obs_number",
        "obs_key",
        "obs_mean",
        "obs_std",
        "status",
        "sim_mean",
        "sim_std",
    ]
    # drop last line ('=====...====')
    # drop last row i.e. end of file line
    updatelog.drop(updatelog.tail(1).index, inplace=True)
    # ---------------------------------
    # add proper name for obs_key in rows where value is '...'
    replace = "..."
    for i in range(len(updatelog.index)):
        obskey_val = updatelog.at[i, "obs_key"]
        if obskey_val == replace:
            updatelog.at[i, "obs_key"] = key

    return updatelog


def make_obs_groups(key_map):
    """Create a mapping of observation groups, the names will be:
    data_key -> [obs_keys] and All_obs-{missing_obs} -> [obs_keys]
    and All_obs -> [all_obs_keys]
    """
    combinations = key_map.copy()
    if len(combinations) == 1:
        return combinations
    # We want all observations and all observations
    # minus individual observations:
    for combination_length in (len(key_map.keys()) - 1, len(key_map.keys())):
        if combination_length == 1:
            # We have hit a case with two observation groups, now we only
            # need individual group and all obs.
            continue
        for subset in itertools.combinations(key_map.keys(), combination_length):
            obs_group = list(
                itertools.chain.from_iterable([key_map[key] for key in subset])
            )
            if combination_length == len(key_map.keys()) - 1:
                missing_obs = [x for x in key_map.keys() if x not in set(subset)]
                assert len(missing_obs) == 1
                name = f"All_obs-{missing_obs[0]}"
            else:
                name = "All_obs"
            combinations[name.replace(":", "_")] = obs_group
    return combinations


def list_active_observations(key, active_obs, df_update_log):
    """To get the inactive vs total observation info."""
    df_active = df_update_log[(df_update_log["status"] == "Active")]
    active_obs.at["ratio", key] = (
        str(len(df_active)) + " active/" + str(len(df_update_log.index))
    )
    return active_obs


def create_path(runpathdf, target_name):
    """to create path for output data"""
    if runpathdf.endswith("/"):
        num = 3
    else:
        num = 2
    newpath = runpathdf.rsplit("/", num)[0]
    # create ouput folder for saving calculated data
    output_path = newpath + "/share/output_analysis/scalar_" + target_name + "/"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        print("Directory ", output_path, " Created ")
    return output_path


def list_observations_misfit(obs_keys, df_update_log, misfit_df):
    """To get the misfit for total observations (active/inactive)."""
    misfit_values = []
    total_obs_nr = len(df_update_log[df_update_log.status.isin(["Active", "Inactive"])])
    for key in obs_keys:
        misfit_name = "MISFIT:" + key
        misfit_values.append(misfit_df[misfit_name].mean())
    return np.mean(misfit_values) / total_obs_nr


def get_field_grid_char(input_grid):
    """Get field grid characteristics/coordinates"""
    input_grid_name = input_grid.rsplit(".", 1)[0]
    try:
        mygrid = Grid(input_grid_name, fformat="eclipserun")
        dataframeg = mygrid.dataframe(activeonly=False)
        return dataframeg
    except OSError as err:
        raise OSError("A grid with .EGRID format is expected.") from err


def calc_delta_grid(
    all_input_post, all_input_prior, mygrid_ok, caseobs, mygrid_ok_short
):
    """calculate mean delta of field grid data"""
    delta_post_prior = np.subtract(all_input_prior, all_input_post)
    delta_post_prior = np.absolute(delta_post_prior)
    mygrid_ok["Mean_delta"] = np.mean(delta_post_prior, axis=0)
    df_mean_delta = mygrid_ok.groupby(["IX", "JY"])[["Mean_delta"]].mean().reset_index()
    mygrid_ok_short["Mean_D_" + caseobs] = df_mean_delta["Mean_delta"]
    return mygrid_ok_short


def get_updated_parameters(prior_data, parameters):
    """make list of updated parameters
    (excluding duplicate transformed parameters)
    """
    transfp_keys = []
    p_keysf = []
    for paramet in parameters:
        for key in prior_data.keys():
            if "_" + paramet + ":" in key:
                transfp_keys.append(key)
    p_keys = set(prior_data.keys()).difference(transfp_keys)
    # remove parameters with constant prior distribution
    for dkey in p_keys:
        if prior_data[dkey].ndim > 1:
            print("WARNING: Parameter " + dkey + " defined several times.")
            flatten_arr = np.ravel(prior_data[dkey])
            result = np.all(prior_data[dkey] == flatten_arr[0])
            if not result:
                p_keysf.append(dkey)
        else:
            if not all(x == prior_data[dkey][0] for x in prior_data[dkey]):
                p_keysf.append(dkey)
    return p_keysf


def calc_ks(ks_matrix, dkeys, prior_data, target_data, key):
    """Calculate KS matrix"""
    for dkey in sorted(dkeys):
        ks_val = ks_2samp(prior_data[dkey], target_data[dkey])
        ks_matrix.at[dkey, key] = ks_val[0]
    return ks_matrix


def check_names(ert_currentname, prior_name, target_name):
    """Check input names given"""
    if prior_name is None:
        prior_name = ert_currentname
    if target_name == "<ANALYSIS_CASE_NAME>":
        target_name = "analysis_case"
    return prior_name, target_name


def check_inputs(misfitdata, prior_data):
    """Check input ensemble prior is not empty
    and if ensemble contains parameters for hm"""
    if misfitdata.empty:
        raise Exception("Empty prior ensemble")
    if prior_data.empty:
        raise Exception("Empty parameters set for History Matching")
    return prior_data
