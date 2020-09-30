from semeio.workflows.correlated_observations_scaling import ObservationScaleFactor
from semeio.workflows.misfit_preprocessor.config import (
    AUTO_SCALE,
    SPEARMAN_CORRELATION,
)
from semeio.workflows.spearman_correlation_job.job import spearman_job


def run(config, measured_data, reporter):

    if config.clustering.method == SPEARMAN_CORRELATION:
        sconfig = config.clustering.spearman_correlation
        scaling_configs = spearman_job(
            measured_data,
            sconfig.fcluster.t,
            reporter,
            criterion=sconfig.fcluster.criterion,
            depth=sconfig.fcluster.depth,
            method=sconfig.linkage.method,
            metric=sconfig.linkage.metric,
        )
    elif config.clustering.method == AUTO_SCALE:
        job = ObservationScaleFactor(reporter, measured_data)
        nr_components, _ = job.perform_pca(config.scaling.threshold)
        sconfig = config.clustering.auto_scale
        scaling_configs = spearman_job(
            measured_data,
            nr_components,
            reporter,
            criterion="maxclust",
            method=sconfig.linkage.method,
            metric=sconfig.linkage.metric,
        )
    else:
        raise AssertionError(
            "Unknown clustering method: {}".format(config.clustering.method)
        )

    return scaling_configs
