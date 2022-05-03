# pylint: disable=unbalanced-tuple-unpacking
import pytest
from sklearn.datasets import make_blobs

from semeio.workflows.spearman_correlation_job.cluster_analysis import (
    fcluster_analysis,
    kmeans_analysis,
)


@pytest.mark.parametrize(
    "func, kwargs",
    ((fcluster_analysis, {"criterion": "maxclust"}), (kmeans_analysis, {})),
)
def test_same_format(func, kwargs):
    # The goal of this test is not to test the actual clustering functions,
    # but rather their format. Scipy clusters (labels) are 1-indexed while
    # sklearn are 0-indexed. We therefore set up a very simple dataset with
    # clearly defined clusters so the result will be the same for all functions.
    features, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.1, random_state=42)
    cluster_result = func(features, 3, **kwargs)
    # The clusters are typically the same, but the labels vary so we perform the
    # simplest test, just checking that the desired labels are present.
    assert set(cluster_result) == {1, 2, 3}
