from semeio.workflows.correlated_observations_scaling import ObservationScaleFactor
from semeio.workflows.misfit_preprocessor.config import (
    AUTO_SCALE,
    SPEARMAN_CORRELATION,
)
from semeio.workflows.spearman_correlation_job.job import spearman_job


def run(config, measured_data, reporter):

    if config.workflow.method == SPEARMAN_CORRELATION:
        sconfig = config.workflow.spearman_correlation.clustering
        scaling_configs = spearman_job(
            measured_data,
            sconfig.hierarchical.t,
            reporter,
            criterion=sconfig.hierarchical.criterion,
            depth=sconfig.hierarchical.depth,
            method=sconfig.hierarchical.method,
            metric=sconfig.hierarchical.metric,
        )
        pca_threshold = config.workflow.spearman_correlation.pca.threshold
    elif config.workflow.method == AUTO_SCALE:
        job = ObservationScaleFactor(reporter, measured_data)
        auto_scale_config = config.workflow.auto_scale
        nr_components, _ = job.perform_pca(auto_scale_config.pca.threshold)
        sconfig = auto_scale_config.clustering
        scaling_configs = spearman_job(
            measured_data,
            nr_components,
            reporter,
            criterion="maxclust",
            method=sconfig.hierarchical.method,
            metric=sconfig.hierarchical.metric,
        )
        pca_threshold = auto_scale_config.pca.threshold
    else:
        raise AssertionError(
            "Unknown clustering method: {}".format(config.workflow.method)
        )

    scaling_params = {"threshold": pca_threshold}
    for scaling_config in scaling_configs:
        scaling_config["CALCULATE_KEYS"].update(scaling_params)

    return scaling_configs
