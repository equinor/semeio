# pylint: disable=logging-fstring-interpolation
import itertools
import logging

import pandas as pd

from semeio.workflows.spearman_correlation_job.cluster_analysis import fcluster_analysis


def spearman_job(
    measured_data,
    reporter,
    cluster_function=fcluster_analysis,
    **cluster_args,
):
    """
    Given measured_data and threshold, it returns configurations that describe
    scaling of these data.
    """
    measured_data.remove_failed_realizations()
    measured_data.remove_inactive_observations()

    simulated_data = measured_data.get_simulated_data()

    correlation_matrix = _calculate_correlation_matrix(simulated_data)
    reporter.publish_csv("correlation_matrix", correlation_matrix)

    clusters = cluster_function(correlation_matrix, **cluster_args)

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
    reporter.publish("clusters", clustered_data)
    job_configs = _config_creation(clustered_data)

    for cluster, val in clustered_data.items():
        logging.info(f"Cluster nr: {cluster}, clustered data: {val}")

    return job_configs


def _cluster_data(data):
    groups = {}
    for (number, key), cluster_group in itertools.groupby(
        sorted(data), key=lambda x: (x[0], x[1])
    ):
        if number not in groups:
            groups[number] = {}
        groups[number].update({key: [index for _, _, index in cluster_group]})
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
            logging.info(f"Removed cluster with singular observation: {cluster}")
    return multiobs_clusters


def _config_creation(clusters):
    config = []
    for cluster in clusters.values():
        config.append(
            {
                "CALCULATE_KEYS": {
                    "keys": [{"key": key, "index": val} for key, val in cluster.items()]
                }
            }
        )
    return config


def _calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    # Spearman correlation is quite slow, but will be improved in a future version
    # of pandas (https://github.com/pandas-dev/pandas/pull/28151), for now this is
    # equivalent:
    return data.rank().corr(method="pearson")
