# -*- coding: utf-8 -*-
import itertools

from ert_data.measured import MeasuredData
from scipy.cluster.hierarchy import linkage, fcluster
from semeio.jobs.correlated_observations_scaling.job import scaling_job


def spearman_job(facade, threshold, dry_run):

    observation_keys = [
        facade.get_observation_key(nr) for nr, _ in enumerate(facade.get_observations())
    ]

    _spearman_correlation(facade, observation_keys, threshold, dry_run)


def _spearman_correlation(facade, obs_keys, threshold, dry_run):
    """
    Collects data, performs scaling and applies scaling, assumes validated input.
    """
    measured_data = MeasuredData(facade, obs_keys)
    measured_data.remove_failed_realizations()
    measured_data.remove_inactive_observations()
    measured_data.filter_ensemble_std(1.0e-6)

    simulated_data = measured_data.get_simulated_data()

    correlation_matrix = _calculate_correlation_matrix(simulated_data)

    clusters = _cluster_analysis(correlation_matrix, threshold)

    columns = correlation_matrix.columns

    # Here the clusters are joined with the key and data index
    # to group the observations, the column level values are the column
    # headers, where key_index is the observation key and data_index
    # is a range.
    data = list(
        zip(
            clusters,
            columns.get_level_values(0),
            columns.get_level_values("data_index"),
        )
    )

    clustered_data = _cluster_data(data)

    job_configs = _config_creation(clustered_data)

    _output_clusters(clustered_data)

    if not dry_run:
        _run_scaling(facade, job_configs)


def _output_clusters(clustered_data):
    for cluster, val in clustered_data.items():
        print("Cluster nr: {}, clustered data: {}".format(cluster, val))


def _run_scaling(facade, job_configs):
    for job in job_configs:
        scaling_job(facade, job)


def _cluster_data(data):
    groups = {}
    for (nr, key), cluster_group in itertools.groupby(
        sorted(data), key=lambda x: (x[0], x[1])
    ):
        if nr not in groups:
            groups[nr] = {}
        groups[nr].update({key: [index for _, _, index in cluster_group]})
    return groups


def _config_creation(clusters):
    config = []
    for cluster_nr, cluster in clusters.items():
        config.append(
            {
                "CALCULATE_KEYS": {
                    "keys": [{"key": key, "index": val} for key, val in cluster.items()]
                }
            }
        )
    return config


def _calculate_correlation_matrix(data):
    # Spearman correlation is quite slow, but will be improved in a future version
    # of pandas (https://github.com/pandas-dev/pandas/pull/28151), for now this is
    # equivalent:
    return data.rank().corr(method="pearson")


def _cluster_analysis(correlation_matrix, threshold):
    a = linkage(correlation_matrix, "single")
    return fcluster(a, threshold)
