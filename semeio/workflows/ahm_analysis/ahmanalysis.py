import argparse
import sys
import os
import re
import itertools
import numpy as np
import pandas as pd
import glob

from scipy.stats.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from res.enkf import ErtScript
from res.enkf import (
    ErtImplType,
    EnkfNode,
    EnKFMain,
)
from res.enkf import (
    ESUpdate,
    ErtRunContext,
    ResConfig,
)
from res.enkf.export import (
    GenKwCollector,
    SummaryObservationCollector,
    GenDataObservationCollector,
    MisfitCollector,
)

from ert_shared.plugins.plugin_manager import ErtPluginContext


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


class AhmAnalysisJob(ErtScript):  # pylint: disable=too-few-public-methods
    """ Define ERT workflow to evaluate change of parameters for each observation"""

    def run(self, target_name="analysis_case", prior_name=None):
        """Perform analysis of parameters change per obs group
        prior to posterior of ahm"""
        ert = self.ert()
        prior_name, target_name = check_names(
            ert.getEnkfFsManager().getCurrentFileSystem().getCaseName(),
            prior_name,
            target_name,
        )
        # Get the prior scalar parameter distributions
        prior_data = check_inputs(
            MisfitCollector.loadAllMisfitData(ert, prior_name),
            GenKwCollector.loadAllGenKwData(ert, prior_name)
        )

        output_path = create_path(
            ert.getModelConfig().getRunpathAsString(), target_name
        )

        gen_obs_list = GenDataObservationCollector.getAllObservationKeys(ert)
        # create dataframe with observations vectors (1 by 1 obs and also all_obs)
        obs_groups = make_obs_groups(
            gen_obs_list + SummaryObservationCollector.getAllObservationKeys(ert),
            ert.getObservations(),
        )
        # setup dataframe for calculated data
        ks_data, active_obs, misfitval = initialize_emptydf()

        field_parameters = sorted(
            ert.ensembleConfig().getKeylistFromImplType(ErtImplType.FIELD)
        )
        scalar_parameters = sorted(
            ert.ensembleConfig().getKeylistFromImplType(ErtImplType.GEN_KW)
        )
        # identify the set of actual parameters that was updated
        # for now just go through scalar parameters 
        # but in future if easier acces to field parameter updates should also include field parameters
        dkeysf = get_updated_parameters(
            prior_data, scalar_parameters
        )

        # loop over keys and calculate the KS metric,
        # conditioning one parameter at the time.
        for key in obs_groups:
            print("Processing:", key)
            # Create a localization scheme

            #    """ Use localization to evaluate change of parameters for each observation"""

            # Reset internal local config structure, in order to make your own
            ert.getLocalConfig().clear()

            # A ministep is used to link betwen data and observations.
            # Make more ministeps to condition different groups together
            ministep = ert.getLocalConfig().createMinistep("MINISTEP")
            # Add all dataset to localize
            data_all = ert.getLocalConfig().createDataset("DATASET")
            for data in field_parameters + scalar_parameters:
                data_all.addNode(data)
            # Add all obs to be used in this updating scheme
            obsdata = ert.getLocalConfig().createObsdata("OBS")
            for obs in obs_groups[key]:
                obsdata.addNode(obs.getObservationKey())
            # Attach the created dataset and obsset to the ministep
            ministep.attachDataset(data_all)
            ministep.attachObsset(obsdata)
            # Then attach the ministep to the update step
            ert.getLocalConfig().getUpdatestep().attachMinistep(ministep)

            # Perform update analysis
            ert.analysisConfig().set_log_path(
                output_path.replace("scalar_", "update_log/")
            )
            run_context = ErtRunContext.ensemble_smoother_update(
                ert.getEnkfFsManager().getFileSystem(prior_name),
                ert.getEnkfFsManager().getFileSystem(target_name),
            )
            ESUpdate(ert).smootherUpdate(run_context)

            # Get the updated scalar parameter distributions

            GenKwCollector.loadAllGenKwData(ert, target_name).to_csv(
                output_path + key + ".csv"
            )

            # Get the inactive vs total observation info
            active_obs = list_active_observations(
                key,
                active_obs,
                output_path,
            )
            # Get misfit values
            misfitval = list_observations_misfit(
                key,
                misfitval,
                gen_obs_list,
                output_path,
                MisfitCollector.loadAllMisfitData(ert, prior_name),
            )
            if field_parameters != []:
                for fieldparam in field_parameters:
                    # Get/export the updated Field parameters
                    out_file = (
                        ert.ensembleConfig()[fieldparam]
                        .get_enkf_outfile()
                        .rsplit(".")[-1]
                    )
                    EnkfNode.exportMany(
                        ert.ensembleConfig()[fieldparam],
                        os.path.join(
                            output_path.replace("scalar_", "field_") + key + "/",
                            "%d_" + fieldparam + "_field." + out_file,
                        ),
                        ert.getEnkfFsManager().getFileSystem(target_name),
                        np.arange(0, ert.getEnkfFsManager().getEnsembleSize()),
                    )

            # Calculate Ks matrix for scalar parameters
            ks_data = calc_ks(
                ks_data,
                dkeysf,
                prior_data,
                GenKwCollector.loadAllGenKwData(ert, target_name),
                key,
            )

        # Calculate Ks matrix for Fields parameters
        if field_parameters != []:
            # Get grid characteristics to be able to plot field avg maps
            mygrid_ok = get_field_grid_char(ert.eclConfig().get_gridfile())

            # Get//Export/import the prior Field parameters
            for fieldparam in field_parameters:
                out_file = (
                    ert.ensembleConfig()[fieldparam].get_enkf_outfile().rsplit(".")[-1]
                )
                EnkfNode.exportMany(
                    ert.ensembleConfig()[fieldparam],
                    os.path.join(
                        output_path.replace("scalar_", "field_") + "prior",
                        "%d_" + fieldparam + "_field." + out_file,
                    ),
                    ert.getEnkfFsManager().getFileSystem(prior_name),
                    np.arange(0, ert.getEnkfFsManager().getEnsembleSize()),
                )
                # Read/import
                all_input_prior = get_input_state_df(
                    out_file,
                    ert.eclConfig().get_gridfile(),
                    ert.getEnkfFsManager().getEnsembleSize(),
                    output_path.replace("scalar_", "field_") + "prior/",
                    fieldparam,
                )
                scaler = StandardScaler()
                scaler.fit(all_input_prior)
                pca = PCA(0.98).fit(pd.DataFrame(scaler.transform(all_input_prior)))
                #                print("Number of PCA:", fieldparam, " ", pca.n_components_)
                pc_fieldprior_df = pd.DataFrame(
                    data=pca.transform(scaler.transform(all_input_prior))
                )
                all_ks = pd.DataFrame()
                # Get the posterior Field parameters
                mygrid_ok_short = mygrid_ok[mygrid_ok["KZ"] == 1].copy().reset_index()
                for key in obs_groups:
                    all_input_post = get_input_state_df(
                        out_file,
                        ert.eclConfig().get_gridfile(),
                        ert.getEnkfFsManager().getEnsembleSize(),
                        output_path.replace("scalar_", "field_") + key + "/",
                        fieldparam,
                    )
                    mygrid_ok_short = calc_delta_grid(
                        all_input_post,
                        all_input_prior,
                        mygrid_ok,
                        key,
                        mygrid_ok_short,
                    )
                    pc_fieldpost_df = pd.DataFrame(
                        data=pca.transform(scaler.transform(all_input_post))
                    )
                    all_ks = calc_ks(
                        all_ks,
                        pc_fieldpost_df,
                        pc_fieldprior_df,
                        pc_fieldpost_df,
                        key,
                    )
                # add the field max Ks to the scalar Ks matrix
                ks_data.loc["FIELD_" + fieldparam] = all_ks.max()
                mygrid_ok_short.to_csv(
                    output_path.replace("scalar_", "field_")
                    + "delta_field"
                    + fieldparam
                    + ".csv"
                )
        # save/export the Ks matrix, active_obs, misfitval and prior data
        save_to_csv(ks_data, active_obs, misfitval, prior_data, output_path)


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
        print("Warning more than one update_log_file.")
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


def make_obs_groups(obs_keys, local_obs):
    """create dataframe with grouped observations
    as appearing in Observations on Ert gui"""
    obs_groups = {}
    for key in obs_keys:
        nkey = re.sub(":", "_", key)
        obs_groups[nkey] = []
    if "All_obs" in obs_groups:
        raise Exception(
            "All_obs observation vector already exists, consider renaming this vector to be able to run the script."
        )
    obs_groups["All_obs"] = []
    for obs in local_obs:
        obs_dkey_o = obs.getDataKey()
        obs_key = obs.getObservationKey()
        obs_dkey = re.sub(":", "_", obs_dkey_o)
        if obs_dkey in obs_groups:
            obs_groups[obs_dkey].append(obs)
        else:
            obs_groups[obs_key].append(obs)
        obs_groups["All_obs"].append(obs)
    inact_obs = 0
    for ggkeys in obs_keys:
        gkeys = re.sub(":", "_", ggkeys)
        obs_groups["All_obs-" + gkeys] = []
        for obs in local_obs:
            obs_dkey_o = obs.getDataKey()
            obs_key = obs.getObservationKey()
            obs_dkey = re.sub(":", "_", obs_dkey_o)
            if gkeys in (obs_dkey, obs_key):
                inact_obs += 1
            else:
                obs_groups["All_obs-" + gkeys].append(obs)
    return obs_groups


# def list_active_observations(key, active_obs, gen_obs_list, output_path, misfit_df):
def list_active_observations(key, active_obs, output_path):
    """To get the inactive vs total observation info."""
    update_log_path = output_path.replace("scalar_", "update_log/")
    df_update_log = make_update_log_df(update_log_path, key)
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


def list_observations_misfit(key, misfitval, gen_obs_list, output_path, misfit_df):
    """To get the misfit for total observations (active/inactive)."""
    update_log_path = output_path.replace("scalar_", "update_log/")
    list_kobs = []
    if key in gen_obs_list:
        misfit_name = "MISFIT:" + key
        if misfit_name not in misfit_df.keys():
            misfitval.at["misfit", key] = "None"
        else:
            df_update_log = make_update_log_df(update_log_path, key)
            df_filter = df_update_log[df_update_log.status.isin(["Active", "Inactive"])]
            if "All_obs" in key:
                misfitval.at["misfit", key] = "None"
            else:
                misfitval.at["misfit", key] = misfit_df[misfit_name].mean() / (
                    len(df_filter.index)
                )
    else:
        df_update_log = make_update_log_df(update_log_path, key)
        df_active = df_update_log[(df_update_log["status"] == "Active")]
        if "All_obs" in key:
            misfitval.at["misfit", key] = "None"
        else:
            for keys_obs in df_active["obs_key"]:
                list_kobs.append("MISFIT:" + keys_obs)
            misfitval.at["misfit", key] = misfit_df[list_kobs].mean(axis=1).mean()
    return misfitval


def get_input_state_df(ext, input_grid, iensnb, df_path_state, fieldparam):
    """create input prior and posterior dataframe"""
    mygrid_char = Grid(input_grid.rsplit(".", 1)[0], fformat="eclipserun")
    all_input = []
    for real in range(iensnb):
        inputdata = df_path_state + str(real) + "_" + fieldparam + "_field." + ext
        array_nb = []
        proproff = GridProperty(inputdata, name=fieldparam, grid=mygrid_char)
        array_nb = proproff.get_npvalues1d(activeonly=False, fill_value=0, order="C")
        all_input.append(array_nb)
    return all_input


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
    p_keys = []
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


def save_to_csv(ks_data, active_obs, misfitval, prior_data, output_path):
    ks_data.to_csv(output_path + "ks.csv")
    active_obs.to_csv(output_path + "active_obs_info.csv")
    misfitval.to_csv(output_path + "misfit_obs_info.csv")
    prior_data.to_csv(output_path + "prior.csv")


def create_parser():
    """Create parser"""
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument(
        "ert_config",
        type=str,
        help="ert config file (could include path to the file)",
    )
    parser.add_argument(
        "--case_update",
        type=str,
        help="name of ert case used for update calculation(default:analysis_case)",
        default="analysis_case",
    )
    parser.add_argument(
        "--case_prior",
        type=str,
        help="name of ert case used as prior (default is ert current_case)",
        default=None,
    )
    return parser
