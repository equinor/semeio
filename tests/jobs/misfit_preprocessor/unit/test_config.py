import pytest

from semeio.workflows import misfit_preprocessor


@pytest.mark.parametrize("workflow", ["spearman_correlation", "auto_scale"])
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
def test_valid_scaling_threshold(threshold, expected_valid, workflow):
    config_data = {"workflow": {workflow: {"pca": {"threshold": threshold}}}}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    if expected_valid:
        assert config.valid, config.errors
    else:
        assert not config.valid
        assert 1 == len(config.errors)
        assert config.errors[0].key_path == (
            "workflow",
            workflow,
            "pca",
            "threshold",
        )


def test_default_scaling_threshold():
    config_data = {}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    assert 0.95 == config.snapshot.workflow.auto_scale.pca.threshold


@pytest.mark.parametrize(
    "clustering_method, expected_valid",
    [
        ("auto_scale", True),
        ("spearman_correlation", True),
        ("super_clustering", False),
        (None, False),
    ],
)
def test_clustering_method(clustering_method, expected_valid):
    config_data = {"workflow": {"method": clustering_method}}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    if expected_valid:
        assert config.valid, config.errors
    else:
        assert not config.valid
        assert 1 == len(config.errors)
        assert ("workflow", "method") == config.errors[0].key_path


def test_default_clustering_method():
    config_data = {}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    assert "auto_scale" == config.snapshot.workflow.method


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
        "workflow": {
            "spearman_correlation": {
                "clustering": {
                    "hierarchical": {
                        "fcluster": {"t": threshold, "criterion": criterion}
                    },
                }
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
        expected_key_path = (
            "workflow",
            "spearman_correlation",
            "clustering",
            "hierarchical",
            "fcluster",
        )
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
        "workflow": {
            "spearman_correlation": {
                "clustering": {
                    "hierarchical": {"fcluster": {"criterion": criterion}},
                }
            },
        },
    }

    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    cluster_config = config.snapshot.workflow.spearman_correlation.clustering
    assert default_threshold == cluster_config.hierarchical.fcluster.t


@pytest.mark.parametrize("workflow", ["spearman_correlation", "auto_scale"])
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
def test_valid_depth(depth, expected_valid, workflow):
    config_data = {
        "workflow": {
            workflow: {
                "clustering": {
                    "hierarchical": {"fcluster": {"depth": depth}},
                }
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
        expected_key_path = (
            "workflow",
            workflow,
            "clustering",
            "hierarchical",
            "fcluster",
            "depth",
        )
        assert expected_key_path == config.errors[0].key_path


def test_default_spearman_depth():
    config_data = {}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    cluster_config = config.snapshot.workflow.spearman_correlation.clustering
    assert 2 == cluster_config.hierarchical.fcluster.depth


@pytest.mark.parametrize("workflow", ["spearman_correlation", "auto_scale"])
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
def test_valid_linkage_method(method, expected_valid, workflow):
    config_data = {
        "workflow": {
            workflow: {"clustering": {"hierarchical": {"linkage": {"method": method}}}}
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
        expected_key_path = (
            "workflow",
            workflow,
            "clustering",
            "hierarchical",
            "linkage",
            "method",
        )
        assert expected_key_path == config.errors[0].key_path


def test_default_linkage_method():
    config_data = {}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    cluster_config = config.snapshot.workflow.spearman_correlation.clustering
    assert "average" == cluster_config.hierarchical.linkage.method


@pytest.mark.parametrize("workflow", ["spearman_correlation", "auto_scale"])
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
def test_valid_linkage_metric(metric, expected_valid, workflow):
    config_data = {
        "workflow": {
            workflow: {"clustering": {"hierarchical": {"linkage": {"metric": metric}}}}
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
        expected_key_path = (
            "workflow",
            workflow,
            "clustering",
            "hierarchical",
            "linkage",
            "metric",
        )
        assert expected_key_path == config.errors[0].key_path


def test_default_linkage_metric():
    config_data = {}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(
        config_data, ("some_observation",)
    )

    assert config.valid, config.errors
    cluster_config = config.snapshot.workflow.spearman_correlation.clustering
    assert "euclidean" == cluster_config.hierarchical.linkage.metric


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


def test_config_workflow():
    config_data = {
        "observations": ["WWPR"],
        "workflow": {
            "spearman_correlation": {
                "clustering": {
                    "hierarchical": {
                        "linkage": {"method": "complete", "metric": "jensenshannon"}
                    }
                },
                "pca": {"threshold": 0.98},
            }
        },
    }
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(config_data, ["WWPR"])
    assert config.valid, config.errors
    assert config.snapshot.workflow.method == "spearman_correlation"


@pytest.mark.parametrize("workflow", [{"spearman_correlation": {}}, {"auto_scale": {}}])
def test_config_workflow_valid(workflow):
    config_data = {"workflow": workflow}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(config_data, ["WWPR"])
    assert config.valid, config.errors
    assert config.snapshot.workflow.method == list(workflow.keys())[0]


@pytest.mark.parametrize(
    "workflow, expected_errors",
    [
        [{"spearman_correlation": {}, "auto_scale": {}}, "Only one entry is expected."],
        [
            {"spearman_correlation": {}, "method": "auto_scale"},
            "<method> is auto_scale but dict_keys(['spearman_correlation', 'method']) "
            "was the one configured.",
        ],
    ],
)
def test_config_workflow_invalid(workflow, expected_errors):
    config_data = {"workflow": workflow}
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(config_data, ["WWPR"])

    expected_msg = (
        f"'Checking that workflow schema only has one method configured"
        f"' failed on input '{workflow}' with error "
    )
    assert not config.valid
    assert expected_msg in config.errors[0].msg
    assert expected_errors in config.errors[0].msg


@pytest.mark.parametrize(
    "key, expected_error",
    [["t", "Unknown key: t"], ["criterion", "Unknown key: criterion"]],
)
def test_auto_scale_invalid_fcluster(key, expected_error):
    config_data = {
        "workflow": {
            "auto_scale": {"clustering": {"hierarchical": {"fcluster": {key: 1}}}}
        }
    }
    config = misfit_preprocessor.config.MisfitPreprocessorConfig(config_data, ["WWPR"])

    assert not config.valid
    assert f"Unknown key: {key}" in config.errors[0].msg
