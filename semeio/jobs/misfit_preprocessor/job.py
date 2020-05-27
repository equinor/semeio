from semeio.jobs.misfit_preprocessor.exceptions import ValidationError
from semeio.jobs.misfit_preprocessor.config import (
    SPEARMAN_CORRELATION,
    assemble_config,
)
from semeio.jobs.spearman_correlation_job.job import spearman_job


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
    else:
        raise AssertionError(
            "Unknown clustering method: {}".format(config.clustering.method)
        )

    return scaling_configs
