from semeio.jobs.misfit_preprocessor.exceptions import ValidationError
from semeio.jobs.correlated_observations_scaling.exceptions import EmptyDatasetException

from semeio.jobs.misfit_preprocessor.config import (
    SPEARMAN_CORRELATION,
    AUTO_CLUSTER,
    assemble_config,
)
from semeio.jobs.spearman_correlation_job.job import spearman_job
from semeio.jobs.scripts.correlated_observations_scaling import (
    CorrelatedObservationsScalingJob,
)


def run(misfit_preprocessor_config, measured_data, reporter):
    config = assemble_config(misfit_preprocessor_config, measured_data)
    if not config.valid:
        raise ValidationError(
            "Invalid configuration of misfit preprocessor", config.errors
        )

    config = config.snapshot
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
    elif config.clustering.method == AUTO_CLUSTER:
        scaling_config = config.scaling
        # call PCA / COS job first and get number of n_components
        try:
            nr_components = CorrelatedObservationsScalingJob.get_nr_primary_components(
                measured_data, scaling_config, reporter
            )[0]
        except IndexError:
            raise AssertionError("An error when acquiring principal components!")
        except EmptyDatasetException:
            pass

        # create as many clusters (or max num clusters) as is n_components with spearmen correlation
        auto_cluster_config = config.clustering.auto_cluster
        scaling_configs = spearman_job(
            measured_data,
            nr_components,  # forming max nr_components clusters
            reporter,
            criterion="maxclust",  # this needs to be maxclust
            depth=auto_cluster_config.fcluster.depth,
            method=auto_cluster_config.linkage.method,
            metric=auto_cluster_config.linkage.metric,
        )
    else:
        raise AssertionError(
            "Unknown clustering method: {}".format(config.clustering.method)
        )

    return scaling_configs
