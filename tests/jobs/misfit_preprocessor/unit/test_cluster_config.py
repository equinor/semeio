import pydantic
import pytest

from semeio.workflows.misfit_preprocessor.hierarchical_config import (
    BaseFclusterConfig,
    FclusterConfig,
    HierarchicalConfig,
    LimitedHierarchicalConfig,
    LinkageConfig,
)


@pytest.mark.parametrize(
    "config_class",
    [
        HierarchicalConfig,
        LinkageConfig,
        BaseFclusterConfig,
        FclusterConfig,
        LimitedHierarchicalConfig,
    ],
)
def test_extra_values(config_class):
    config_data = {"not_a_key": "not_a_val"}
    with pytest.raises(pydantic.ValidationError, match="extra fields not permitted"):
        config_class(**config_data)
