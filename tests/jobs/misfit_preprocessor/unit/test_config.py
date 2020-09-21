import pytest

from semeio.workflows import misfit_preprocessor


@pytest.mark.parametrize(
    "threshold, expected_valid",
    [
        (-1, False),
        (0, False),
        (10, False),
        (1.0001, False),
        (0.001, True),
        (0.5, True),
        (0.999, True),
    ],
)
def test_valid_scaling_threshold(threshold, expected_valid):
    config_data = {
        "scaling": {"threshold": threshold, "std_cutoff": 2, "alpha": 3},
    }
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    if expected_valid:
        assert config.valid, config.errors
    else:
        assert not config.valid
        assert 1 == len(config.errors)
        assert ("scaling", "threshold") == config.errors[0].key_path


def test_default_scaling_threshold():
    config_data = {
        "scaling": {"std_cutoff": 2, "alpha": 3},
    }
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    assert 0.95 == config.snapshot.scaling.threshold


@pytest.mark.parametrize(
    "std_cutoff, expected_valid", [(1, True), (2, True), (None, True)]
)
def test_scaling_std_cutoff(std_cutoff, expected_valid):
    config_data = {
        "scaling": {"threshold": 0.1, "std_cutoff": std_cutoff, "alpha": 3},
    }
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    if expected_valid:
        assert config.valid, config.errors
    else:
        assert not config.valid
        assert 1 == len(config.errors)
        assert ("scaling", "std_cutoff") == config.errors[0].key_path


@pytest.mark.parametrize("alpha, expected_valid", [(1, True), (2, True)])
def test_scaling_alpha(alpha, expected_valid):
    config_data = {
        "scaling": {"alpha": alpha},
    }
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    if expected_valid:
        assert config.valid, config.errors
    else:
        assert not config.valid
        assert 1 == len(config.errors)
        assert ("scaling", "alpha") == config.errors[0].key_path


@pytest.mark.parametrize(
    "clustering_method, expected_valid",
    [("spearman_correlation", True), ("super_clustering", False), (None, False)],
)
def test_clustering_method(clustering_method, expected_valid):
    config_data = {"clustering": {"method": clustering_method}}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    if expected_valid:
        assert config.valid, config.errors
    else:
        assert not config.valid
        assert 1 == len(config.errors)
        assert ("clustering", "method") == config.errors[0].key_path


def test_default_clustering_method():
    config_data = {}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    assert "auto_scale" == config.snapshot.clustering.method


@pytest.mark.parametrize(
    "threshold, criterion, expected_valid, postfix_key_path",
    [
        (None, "inconsistent", False, ("t",)),
        (None, "distance", False, ("t",)),
        (None, "maxclust", False, ("t",)),
        (None, "monocrit", False, ("t",)),
        (None, "maxclust_monocrit", False, ("t",)),
        (0, "inconsistent", False, ("t",)),
        (0, "distance", False, ("t",)),
        (0, "maxclust", False, ("t",)),
        (0, "monocrit", False, ("t",)),
        (0, "maxclust_monocrit", False, ("t",)),
        (12, "inconsistent", True, ()),
        (12, "distance", True, ()),
        (12, "maxclust", True, ()),
        (12, "monocrit", True, ()),
        (12, "maxclust_monocrit", True, ()),
        (1.2, "inconsistent", True, ()),
        (1.2, "distance", True, ()),
        (1.2, "maxclust", False, ()),
        (1.2, "monocrit", True, ()),
        (1.2, "maxclust_monocrit", False, ()),
    ],
)
def test_valid_spearman_threshold(
    threshold, criterion, expected_valid, postfix_key_path
):
    config_data = {
        "clustering": {
            "spearman_correlation": {
                "fcluster": {"t": threshold, "criterion": criterion},
            },
        },
    }
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    if expected_valid:
        assert config.valid, config.errors
    else:
        assert not config.valid
        assert 1 == len(config.errors)
        expected_key_path = ("clustering", "spearman_correlation", "fcluster")
        expected_key_path += postfix_key_path
        assert expected_key_path == config.errors[0].key_path


@pytest.mark.parametrize(
    "criterion, default_threshold",
    [
        ("inconsistent", 1.15),
        ("distance", 1.15),
        ("maxclust", 5),
        ("monocrit", 1.15),
        ("maxclust_monocrit", 5),
    ],
)
def test_default_spearman_threshold(criterion, default_threshold):
    config_data = {
        "clustering": {"spearman_correlation": {"fcluster": {"criterion": criterion}}}
    }
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    assert (
        default_threshold == config.snapshot.clustering.spearman_correlation.fcluster.t
    )


@pytest.mark.parametrize(
    "depth, expected_valid",
    [
        (-10, False),
        (0, False),
        (1, True),
        (10 ** 10, True),
        (None, False),
        (1.5, False),
    ],
)
def test_valid_spearman_depth(depth, expected_valid):
    config_data = {
        "clustering": {"spearman_correlation": {"fcluster": {"depth": depth}}},
    }
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    if expected_valid:
        assert config.valid, config.errors
    else:
        assert not config.valid
        assert 1 == len(config.errors)
        expected_key_path = ("clustering", "spearman_correlation", "fcluster", "depth")
        assert expected_key_path == config.errors[0].key_path


def test_default_spearman_depth():
    config_data = {}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    assert 2 == config.snapshot.clustering.spearman_correlation.fcluster.depth


@pytest.mark.parametrize(
    "method, expected_valid",
    [
        ("single", True),
        ("complete", True),
        ("average", True),
        ("weighted", True),
        ("centroid", True),
        ("ward", True),
        (None, False),
        ("secret", False),
    ],
)
def test_valid_linkage_method(method, expected_valid):
    config_data = {
        "clustering": {"spearman_correlation": {"linkage": {"method": method}}},
    }
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    if expected_valid:
        assert config.valid, config.errors
    else:
        assert not config.valid
        assert 1 == len(config.errors)
        expected_key_path = ("clustering", "spearman_correlation", "linkage", "method")
        assert expected_key_path == config.errors[0].key_path


def test_default_linkage_method():
    config_data = {}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    assert "average" == config.snapshot.clustering.spearman_correlation.linkage.method


@pytest.mark.parametrize(
    "metric, expected_valid",
    [
        ("braycurtis", True),
        ("canberra", True),
        ("chebyshev", True),
        ("cityblock", True),
        ("correlation", True),
        ("cosine", True),
        ("dice", True),
        ("euclidean", True),
        ("hamming", True),
        ("jaccard", True),
        ("jensenshannon", True),
        ("kulsinski", True),
        ("mahalanobis", True),
        ("matching", True),
        ("minkowski", True),
        ("rogerstanimoto", True),
        ("russellrao", True),
        ("seuclidean", True),
        ("sokalmichener", True),
        ("sokalsneath", True),
        ("sqeuclidean", True),
        ("yule", True),
        (None, False),
        ("secret", False),
    ],
)
def test_valid_linkage_metric(metric, expected_valid):
    config_data = {
        "clustering": {"spearman_correlation": {"linkage": {"metric": metric}}},
    }
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    if expected_valid:
        assert config.valid, config.errors
    else:
        assert not config.valid
        assert 1 == len(config.errors)
        expected_key_path = ("clustering", "spearman_correlation", "linkage", "metric")
        assert expected_key_path == config.errors[0].key_path


def test_default_linkage_metric():
    config_data = {}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    assert "euclidean" == config.snapshot.clustering.spearman_correlation.linkage.metric


@pytest.mark.parametrize(
    "observation_filter, observation_keys, expected_obs, expected_valid",
    [
        (("my_obs",), ("my_obs", "other_obs"), ("my_obs",), True),
        (("my_obs", "obs1"), ("my_obs", "other_obs", "obs1"), ("my_obs", "obs1"), True),
        (("secret_obs",), ("my_obs", "other_obs"), (), False),
        (("*",), ("my_obs", "other_obs"), ("my_obs", "other_obs"), True),
        (
            ("*1", "*2"),
            ("a1", "a2", "a3", "aa1", "aa2", "aa3"),
            ("a1", "a2", "aa1", "aa2"),
            True,
        ),
        (
            ("*1", "*2", "*5"),
            ("a1", "a2", "a3", "aa1", "aa2", "aa3"),
            ("a1", "a2", "aa1", "aa2"),
            False,
        ),
    ],
)
def test_valid_observations(
    observation_filter,
    observation_keys,
    expected_obs,
    expected_valid,
):
    config_data = {"observations": observation_filter}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, observation_keys
    )

    if expected_valid:
        assert config.valid, config.errors
        assert sorted(config.snapshot.observations) == sorted(expected_obs)
    else:
        assert not config.valid
        assert 1 == len(config.errors)
        expected_key_path = ("observations",)
        assert expected_key_path == config.errors[0].key_path[:-1]


@pytest.mark.parametrize(
    "observation_keys",
    [
        ("my_obs", "other_obs"),
        ("my_obs", "other_obs", "obs1") + tuple(i * "a" for i in range(20)),
        ("my_obs", "other_obs"),
    ],
)
def test_default_observations(observation_keys):
    config_data = {}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, observation_keys
    )

    assert config.valid, config.errors
    assert sorted(observation_keys) == sorted(config.snapshot.observations)
