import pytest

from semeio.workflows.misfit_preprocessor import workflow_config


@pytest.mark.parametrize(
    "key, value",
    [["threshold", 0.010], ["criterion", "distance"]],
)
def test_union_select_spearman(key, value):
    config = workflow_config.MisfitConfig(
        workflow={"clustering": {"fcluster": {key: value}}}
    )
    assert config.workflow.type == "custom_scale"


def test_union_select_auto_scale():
    config = workflow_config.MisfitConfig(
        workflow={"clustering": {"fcluster": {"depth": 1}}}
    )
    assert config.workflow.type == "auto_scale"


@pytest.mark.parametrize(
    "key, value",
    [["threshold", 0.010], ["criterion", "distance"], ["depth", 10]],
)
def test_custom_scale_valid_fcluster(key, value):
    config_data = {
        "type": "custom_scale",
        "clustering": {"fcluster": {key: value}},
    }
    workflow_config.CustomScaleConfig(**config_data)


@pytest.mark.parametrize(
    "key, value",
    [["threshold", 0.010], ["criterion", "distance"]],
)
def test_workflow_custom(key, value):
    config_data = {
        "type": "custom_scale",
        "clustering": {"fcluster": {key: value}},
    }
    workflow_config.MisfitConfig(workflow=config_data)
