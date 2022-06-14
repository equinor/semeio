from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans


def fcluster_analysis(
    correlation_matrix,
    threshold=1.0,
    criterion="inconsistent",
    depth=2,
    method="single",
    metric="euclidean",
):
    # pylint: disable=too-many-arguments
    a = linkage(correlation_matrix, method, metric)
    return fcluster(a, threshold, criterion=criterion, depth=depth)


def kmeans_analysis(
    correlation_matrix,
    n_clusters=8,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=None,
):
    # pylint: disable=too-many-arguments
    kmeans = KMeans(
        init=init,
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    ).fit(correlation_matrix)
    return kmeans.labels_ + 1  # Scikit clusters are 0-indexed while scipy is 1-indexed
