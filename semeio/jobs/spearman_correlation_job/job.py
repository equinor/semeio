# -*- coding: utf-8 -*-
import itertools

from scipy.cluster.hierarchy import fcluster, linkage


def spearman_job(measured_data, threshold):
    """
    Given measured_data and threshold, it returns configurations that describe
    scaling of these data.
    """
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

    clustered_data = _remove_singular_obs(_cluster_data(data))

    job_configs = _config_creation(clustered_data)

    for cluster, val in clustered_data.items():
        print("Cluster nr: {}, clustered data: {}".format(cluster, val))

    return job_configs


def _cluster_data(data):
    groups = {}
    for (nr, key), cluster_group in itertools.groupby(
        sorted(data), key=lambda x: (x[0], x[1])
    ):
        if nr not in groups:
            groups[nr] = {}
        groups[nr].update({key: [index for _, _, index in cluster_group]})
    return groups


def _remove_singular_obs(clusters):
    """Removes clusters with a singular observation."""
    new_cluster_index = 0
    multiobs_clusters = {}
    for _, cluster in clusters.items():
        if sum(map(len, cluster.values())) > 1:
            multiobs_clusters[new_cluster_index] = cluster
            new_cluster_index += 1
        else:
            print("Removed cluster with singular observation: {}".format(cluster))
    return multiobs_clusters


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
