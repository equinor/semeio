from semeio.workflows.correlated_observations_scaling import ObservationScaleFactor
from semeio.workflows.spearman_correlation_job.job import spearman_job


def run(config, measured_data, reporter):
    workflow = config.workflow
    if workflow.type == "custom_scale":
        sconfig = workflow.clustering.cluster_args()
        scaling_configs = spearman_job(
            measured_data,
            reporter,
            **sconfig,
        )
        pca_threshold = config.workflow.pca.threshold
    elif workflow.type == "auto_scale":
        job = ObservationScaleFactor(reporter, measured_data)
        nr_components, _ = job.perform_pca(workflow.pca.threshold)
        sconfig = workflow.clustering.cluster_args()
        scaling_configs = spearman_job(
            measured_data,
            reporter,
            criterion="maxclust",
            threshold=nr_components,
            **sconfig,
        )
        pca_threshold = workflow.pca.threshold
    else:
        raise AssertionError(
            "Unknown clustering method: {}".format(config.workflow.method)
        )

    scaling_params = {"threshold": pca_threshold}
    for scaling_config in scaling_configs:
        scaling_config["CALCULATE_KEYS"].update(scaling_params)

    return scaling_configs
