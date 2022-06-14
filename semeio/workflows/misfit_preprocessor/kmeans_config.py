from typing import Literal

from pydantic import Field, PrivateAttr, PyObject, conint

from semeio.workflows.misfit_preprocessor.hierarchical_config import (
    AbstractClusteringConfig,
)
from semeio.workflows.spearman_correlation_job.cluster_analysis import kmeans_analysis


class LimitedKmeansClustering(AbstractClusteringConfig):
    """
    The kmeans implementation is backed by sklearn and for a
    more detailed description we refer the reader to the
    documentation of sklearn.cluster.KMeans
    (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
    """

    type: Literal["limited_kmeans"] = "limited_kmeans"
    init: Literal["k-means++", "random"] = Field(
        "k-means++",
        description=("The criterion to use in forming flat clusters. "),
    )
    n_init: conint(gt=0, strict=True) = Field(
        10,
        description=(
            "Number of time the k-means algorithm will be run with "
            "different centroid seeds. The final results will be the "
            "best output of n_init consecutive runs in terms of inertia."
        ),
    )
    max_iter: conint(gt=0, strict=True) = Field(
        300,
        description=(
            "Maximum number of iterations of the k-means algorithm for a single run."
        ),
    )
    random_state: conint(gt=0, strict=True) = Field(
        None,
        description=(
            "Determines random number generation for centroid initialization. "
            "Use an int to make the randomness deterministic"
        ),
    )
    _cluster_function: PyObject = PrivateAttr(kmeans_analysis)


class KmeansClustering(LimitedKmeansClustering):
    __doc__ = LimitedKmeansClustering.__doc__
    type: Literal["kmeans"] = "kmeans"
    n_clusters: conint(gt=0, strict=True) = Field(
        8,
        description=(
            "The scalar gives the maximum number of clusters to be formed."
            "This has a defaulted value for auto_scale "
            "and is not configurable for this workflow. "
        ),
    )
