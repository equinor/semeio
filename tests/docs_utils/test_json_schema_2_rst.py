import pytest
import rstcheck_core.checker

from semeio._docs_utils._json_schema_2_rst import (
    _create_docs,
    _remove_key,
    _replace_key,
)

json_schema = {
    "title": "MisfitConfig",
    "description": "Super nice description!\nWith multiple lines!",
    "type": "object",
    "properties": {
        "workflow": {
            "title": "Workflow",
            "default": {
                "type": "auto_scale",
                "clustering": {
                    "type": "limited_hierarchical",
                    "linkage": {"method": "average", "metric": "euclidean"},
                    "fcluster": {"depth": 2},
                },
                "pca": {"threshold": 0.95},
            },
            "allOf": [{"$ref": "AutoScaleConfig"}],
        },
        "observations": {
            "title": "Observations",
            "default": [],
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "additionalProperties": False,
    "definitions": {
        "LinkageConfig": {
            "title": "LinkageConfig",
            "description": "Short description",
            "type": "object",
            "properties": {
                "method": {
                    "title": "Method",
                    "default": "average",
                    "enum": [
                        "single",
                        "complete",
                        "average",
                        "weighted",
                        "centroid",
                        "ward",
                    ],
                    "type": "string",
                },
                "metric": {
                    "title": "Metric",
                    "default": "euclidean",
                    "enum": [
                        "sokalsneath",
                        "sqeuclidean",
                        "yule",
                    ],
                    "type": "string",
                },
            },
            "additionalProperties": False,
        },
        "BaseFclusterConfig": {
            "title": "BaseFclusterConfig",
            "description": "Description",
            "type": "object",
            "properties": {
                "depth": {
                    "title": "Depth",
                    "default": 2,
                    "exclusiveMinimum": 0,
                    "type": "integer",
                }
            },
            "additionalProperties": False,
        },
        "LimitedHierarchicalConfig": {
            "title": "LimitedHierarchicalConfig",
            "type": "object",
            "properties": {
                "type": {
                    "title": "Type",
                    "default": "limited_hierarchical",
                    "enum": ["limited_hierarchical"],
                    "type": "string",
                },
                "linkage": {
                    "title": "Linkage",
                    "default": {"method": "average", "metric": "euclidean"},
                    "allOf": [{"$ref": "LinkageConfig"}],
                },
                "fcluster": {
                    "title": "Fcluster",
                    "default": {"depth": 2},
                    "allOf": [{"$ref": "BaseFclusterConfig"}],
                },
            },
            "additionalProperties": False,
        },
        "PCAConfig": {
            "title": "PCAConfig",
            "type": "object",
            "properties": {
                "threshold": {
                    "title": "Threshold",
                    "default": 0.95,
                    "exclusiveMinimum": 0,
                    "maximum": 1.0,
                    "type": "number",
                }
            },
            "additionalProperties": False,
        },
        "AutoScaleConfig": {
            "title": "AutoScaleConfig",
            "type": "object",
            "properties": {
                "type": {
                    "title": "Type",
                    "default": "auto_scale",
                    "enum": ["auto_scale"],
                    "type": "string",
                },
                "clustering": {
                    "title": "Clustering",
                    "default": {
                        "type": "limited_hierarchical",
                        "linkage": {"method": "average", "metric": "euclidean"},
                        "fcluster": {"depth": 2},
                    },
                    "allOf": [{"$ref": "LimitedHierarchicalConfig"}],
                },
                "pca": {
                    "title": "Pca",
                    "default": {"threshold": 0.95},
                    "allOf": [{"$ref": "PCAConfig"}],
                },
            },
            "additionalProperties": False,
        },
    },
}


def test_json_2_rst():
    result = _create_docs(json_schema)
    assert not list(rstcheck_core.checker.check_source(result))


@pytest.mark.parametrize(
    "input_dict, old_key, new_key, expected_result",
    [
        ({"a": 1}, "a", "b", {"b": 1}),
        ({"a": 1}, "a", "a", {"a": 1}),
        ({"a": 1}, "b", "c", {"a": 1}),
        ({"a": {"b": 1}}, "b", "c", {"a": {"c": 1}}),
        ({"b": {"b": 1}}, "b", "c", {"c": {"c": 1}}),
        ({"b": {"b": {"b": 1}}}, "b", "c", {"c": {"c": {"c": 1}}}),
        ({"b": {"a": {"b": 1}}}, "b", "c", {"c": {"a": {"c": 1}}}),
    ],
)
def test_replace_key(input_dict, old_key, new_key, expected_result):
    _replace_key(input_dict, old_key, new_key)
    assert input_dict == expected_result


@pytest.mark.parametrize(
    "input_dict, remove_key, expected_result",
    [
        ({"a": 1}, "b", {"a": 1}),
        ({"a": 1}, "a", {}),
        ({"a": {"b": 1}}, "b", {"a": {}}),
        ({"a": {"b": {"c": 1, "d": 2}}}, "c", {"a": {"b": {"d": 2}}}),
    ],
)
def test_remove_key(input_dict, remove_key, expected_result):
    _remove_key(input_dict, remove_key)
    assert input_dict == expected_result
