# pylint: disable=E0213
import itertools
import pathlib

from typing import List, Optional, Union, Dict
from typing_extensions import Literal
from pydantic import BaseModel as PydanticBaseModel
from pydantic import validator, confloat, conint, conlist, root_validator, Extra

from semeio.workflows.localisation.localisation_debug_settings import (
    LogLevel,
)


def expand_wildcards(patterns, list_of_words):
    all_matches = []
    errors = []
    for pattern in patterns:
        matches = [
            words for words in list_of_words if pathlib.Path(words).match(pattern)
        ]
        if len(matches) > 0:
            all_matches.extend(matches)
        else:
            errors.append(f"No match for: {pattern}")
    all_matches = set(all_matches)
    if len(errors) > 0:
        raise ValueError(
            " These specifications does not match anything defined in ERT model\n"
            f"     {errors}, available: {list_of_words}"
        )
    return all_matches


def check_for_duplicated_correlation_specifications(correlations):
    # All observations and model parameters used in correlations
    all_combinations = []

    for corr in correlations:
        all_combinations.extend(
            list(
                itertools.product(
                    corr.obs_group.result_items, corr.param_group.result_items
                )
            )
        )
    errors = []
    seen = set()
    for combination in all_combinations:
        if combination in seen:
            errors.append(f"Observation: {combination[0]}, parameter: {combination[1]}")
        else:
            seen.add(combination)
    return errors


class BaseModel(PydanticBaseModel):
    # pylint: disable=R0903
    class Config:
        extra = Extra.forbid


class ObsConfig(BaseModel):
    """
    Specification of list of observations. A wildcard notation is allowed.
    Use  the 'add' keyword to specify observations to include,
    and the 'remove' keyword to remove some of the observations
    specified by the 'add' keyword. The 'remove' keyword is useful when
    observations are specified by wildcard notation.
    Example:
    obs_group:
         add:  ["WELLA_WWCT*", "WELLB_FOPR", "WELLC*_WOPR*"]
         remove: ["WELLC2*"]
    """

    add: Union[str, List[str]]
    remove: Optional[Union[str, List[str]]]
    context: List[str]
    result_items: Optional[List[str]]

    @validator("add")
    def validate_add(cls, add):
        if isinstance(add, str):
            add = [add]
        return add

    @validator("remove")
    def validate_remove(cls, remove):
        if isinstance(remove, str):
            remove = [remove]
        return remove

    @validator("result_items", always=True)
    def expanded_items(cls, _, values):
        add, remove = values["add"], values.get("remove", None)
        result = _check_specification(add, remove, values["context"])
        if len(result) == 0:
            raise ValueError(
                f"Adding: {add} and removing: {remove} resulted in no items"
            )
        return result


class ParamConfig(ObsConfig):
    """
    Specification of list of model parameters. Wildcard notation is allowed.
    The syntax and meaning of  the 'add' and 'remove' keywords are the
    same as for specification of observations.

    For nodes with parameters defined by the ERT keyword GEN_KW, the
    parameter name is specified on the format: <NODENAME>:<PARAMETER_NAME>
    For nodes with parameters defined by the ERT keyword GEN_PARAM, the
    parameter name is specified on the format: <NODENAME>:<index>.
    For nodes with parameters defined by the ERT keyword SURFACE and FIELD,
    the parameters belonging to a surface node is specified by: <NODENAME>

    Example:
    param_group:
         add: ["GEO_PARAM:*", "FIELD_PARAM*", "GENERAL:*", "SURF_PARAM_*"]
         remove: ["FIELD_PARAM_GRF1", "GEO_PARAM:A*", "GENERAL:2", "GENERAL:3"]

    (This includes all model parameters from node GEO_PARAM except parameter A,
      all field parameter nodes starting with node name FIELD_PARAM except
      FIELD_PARAM_GRF1,
      all surface parameter nodes starting with node name SURF_PARAM_,
      all parameters for node name GENERAL of type GEN_PARAM
      except parameters with indices 2 and 3.)
    """


class GaussianConfig(BaseModel):
    """
    Method for calculating correlation scaling factor using a Gaussian function
    centered at the specified reference point. The ranges and orientation define
    how the scaling factor is reduced away from the reference point.

    This method can be used both to scale correlations for model parameter
    nodes of type FIELD and SURFACE. For nodes of type surface,
    a file specifying the grid layout of the surface must be specified.
    """

    method: Literal["gaussian_decay"]
    main_range: confloat(gt=0)
    perp_range: confloat(gt=0)
    azimuth: confloat(ge=0.0, le=360)
    surface_file: Optional[str]


class ExponentialConfig(GaussianConfig):
    """
    Method for calculating correlation scaling factor using Exponential function.
    See the doc string for Gaussian function for more details.
    """

    method: Literal["exponential_decay"]


class ScalingFromFileConfig(BaseModel):
    method: Literal["from_file"]
    filename: str
    param_name: str


class ScalingForSegmentsConfig(BaseModel):
    method: Literal["segment"]
    segment_filename: str
    param_name: str
    active_segments: Union[conint(ge=0), conlist(item_type=conint(ge=0), min_items=1)]
    scalingfactors: Union[
        confloat(ge=0), conlist(item_type=confloat(ge=0, le=1), min_items=1)
    ]
    smooth_ranges: Optional[
        conlist(item_type=conint(ge=0), min_items=2, max_items=2)
    ] = [0, 0]

    @validator("scalingfactors")
    def check_length_consistency(cls, v, values):
        # Ensure that active segment list and scaling factor lists are of equal length.
        scalingfactors = v
        active_segment_list = values.get("active_segments", None)
        if not active_segment_list:
            return scalingfactors
        if len(scalingfactors) != len(active_segment_list):
            raise ValueError(
                "The specified length of 'active_segments' list"
                f"{active_segment_list }\n"
                f"  and 'scalingfactors' list {scalingfactors} are different."
            )
        return scalingfactors


class CorrelationConfig(BaseModel):
    """
    The keyword 'correlations' specify a set of observations and model parameters
    to have active correlations. For scalar parameters coming from ERT keyword GEN_KW
    and GEN_PARAM , the correlations estimated by ERT will not be reduced,
    and the scaling factor is 1.

    For model parameter nodes of type FIELD or SURFACE, additional
    information about how to adjust the correlations can be specified. The sub-keywords
    'field_scale' and 'surface_scale' are used for nodes of type FIELD or
    SURFACE respectively. It is possible to not specify these keywords, and in that
    case it means that ERT estimated correlations between any field model parameter,
    e.g the field model parameter belonging to a grid cell (i,j,k) and the observations
    specified, is not modified and the scaling factor is 1. If on the other hand,
    a specification of a method for calculating scaling factor exits,
    the covariance will be reduced by this factor:

    new_cov(field(i,j,k)) = orig_cov(field(i,j,k), obs) * scaling_factor

    For some of methods for calculating scaling factors, a reference point,
    typically a well location, must be specified. For other methods any reference
    point is not used.

    """

    name: str
    obs_group: ObsConfig
    param_group: ParamConfig
    ref_point: Optional[conlist(float, min_items=2, max_items=2)]
    field_scale: Optional[
        Union[
            GaussianConfig,
            ExponentialConfig,
            ScalingFromFileConfig,
            ScalingForSegmentsConfig,
        ]
    ]
    surface_scale: Optional[Union[GaussianConfig, ExponentialConfig]]
    obs_context: list
    params_context: list

    @root_validator(pre=True)
    def inject_context(cls, values: Dict) -> Dict:
        values["obs_group"]["context"] = values["obs_context"]
        values["param_group"]["context"] = values["params_context"]
        return values

    @validator("field_scale", pre=True)
    def validate_field_scale(cls, value):
        """
        To improve the user feedback we explicitly check
        which method is configured and bypass the Union
        """
        if isinstance(value, BaseModel):
            return value
        if not isinstance(value, dict):
            raise ValueError("value must be dict")
        method = value.get("method")
        _valid_methods = {
            "gaussian_decay": GaussianConfig,
            "exponential_decay": ExponentialConfig,
            "from_file": ScalingFromFileConfig,
            "segment": ScalingForSegmentsConfig,
        }
        if method in _valid_methods:
            return _valid_methods[method](**value)
        else:
            valid_list = list(_valid_methods.keys())
            raise ValueError(
                f"Unknown method: {method}, valid methods are: {valid_list}"
            )

    @root_validator()
    def valid_ref_point_and_scale(cls, values: Dict) -> Dict:
        field_scale = values.get("field_scale", None)
        surface_scale = values.get("surface_scale", None)
        ref_point = values.get("ref_point")
        if field_scale is not None:
            # ref_point is required for method:
            # - gaussian_decay
            # - exponential_decay
            if field_scale.method in ["gaussian_decay", "exponential_decay"]:
                if ref_point is None:
                    raise ValueError(
                        "When using FIELD with scaling of correlation with "
                        f"method {field_scale.method}, "
                        "the reference point must be specified."
                    )
        if surface_scale is not None:
            # ref_point is required for method:
            # - gaussian_decay
            # - exponential_decay
            if surface_scale.method in ["gaussian_decay", "exponential_decay"]:
                if ref_point is None:
                    raise ValueError(
                        "When using SURFACE with scaling of correlation with "
                        f"method {surface_scale.method}, "
                        "the reference point must be specified."
                    )

        return values

    @validator("surface_scale", pre=True)
    def validate_surface_scale(cls, value):
        """
        To improve the user feedback we explicitly check
        which method is configured and bypass the Union
        """
        if isinstance(value, BaseModel):
            return value
        if not isinstance(value, dict):
            raise ValueError("value must be dict")

        # String with relative path to surface files relative to config path
        key = "surface_file"
        if key not in value.keys():
            raise ValueError(f"Missing keyword: '{key}' in keyword: 'surface_scale' ")

        filename = pathlib.Path(value[key])
        # Check  that the file exists
        if not filename.exists():
            raise ValueError(f"File for surface: {filename} does not exist.")

        method = value.get("method")
        _valid_methods = {
            "gaussian_decay": GaussianConfig,
            "exponential_decay": ExponentialConfig,
        }

        if method in _valid_methods:
            return _valid_methods[method](**value)
        else:
            valid_list = list(_valid_methods.keys())
            raise ValueError(
                f"Unknown method: {method}, valid methods are: {valid_list}"
            )


class LocalisationConfig(BaseModel):
    """
    observations:  A list of observations from ERT in format nodename
    parameters:    A dict of  parameters from ERT in format nodename:paramname.
                            Key is node name. Values are lists of parameter names
                            for the node.
    correlations:   A list of CorrelationConfig objects keeping name of
                            one correlation set which defines the input to
                            create a ministep object.
    log_level:       Integer defining how much log output to write to screen
    write_scaling_factors: Turn on writing calculated scaling parameters to file.
                            Possible values: True/False. Default: False
    """

    observations: List[str]
    parameters: List[str]
    correlations: List[CorrelationConfig]
    log_level: Optional[conint(ge=0, le=5)] = 1
    write_scaling_factors: Optional[bool] = False

    @validator("log_level")
    def validate_log_level(cls, level):
        if not isinstance(level, int):
            level = LogLevel.OFF
        return level

    @root_validator(pre=True)
    def inject_context(cls, values: Dict) -> Dict:
        for correlation in values["correlations"]:
            correlation["obs_context"] = values["observations"]
            correlation["params_context"] = values["parameters"]
        return values

    @validator("correlations")
    def validate_correlations(cls, correlations):
        duplicates = check_for_duplicated_correlation_specifications(correlations)
        if len(duplicates) > 0:
            error_msgs = "\n".join(duplicates)
            raise ValueError(
                f"Found {len(duplicates)} duplicated correlations: \n{error_msgs}"
            )
        return correlations


def _check_specification(items_to_add, items_to_remove, valid_items):
    added_items = expand_wildcards(items_to_add, valid_items)
    if items_to_remove is not None:
        removed_items = expand_wildcards(items_to_remove, valid_items)
        added_items = added_items.difference(removed_items)
    added_items = list(added_items)
    return sorted(added_items)
