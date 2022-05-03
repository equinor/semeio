from semeio.workflows.correlated_observations_scaling import ObservationScaleFactor
from semeio.workflows.spearman_correlation_job.job import spearman_job


def run(config, measured_data, reporter):
    workflow = config.workflow
    sconfig = workflow.clustering.cluster_args()
    if workflow.type == "auto_scale":
        job = ObservationScaleFactor(reporter, measured_data)
        nr_components, _ = job.perform_pca(workflow.pca.threshold)
        if workflow.clustering.type == "limited_hierarchical":
            sconfig["criterion"] = "maxclust"
            sconfig["threshold"] = nr_components
        elif workflow.clustering.type == "limited_kmeans":
            sconfig["n_clusters"] = nr_components

    # pylint: disable=protected-access
    scaling_configs = spearman_job(
        measured_data,
        reporter,
        cluster_function=workflow.clustering._cluster_function,
        **sconfig,
    )
    pca_threshold = workflow.pca.threshold

    scaling_params = {"threshold": pca_threshold}
    for scaling_config in scaling_configs:
        scaling_config["CALCULATE_KEYS"].update(scaling_params)

    return scaling_configs
