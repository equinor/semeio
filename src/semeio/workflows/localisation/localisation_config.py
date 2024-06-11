import itertools
import pathlib
from typing import List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    confloat,
    conint,
    conlist,
    field_validator,
)
from typing_extensions import Annotated


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

    model_config = ConfigDict(extra="forbid")

    add: Union[str, List[str]]
    remove: Optional[Union[str, List[str]]] = None

    @computed_field  # type: ignore[misc]
    @property
    def result_items(self) -> List[str]:
        res = _check_specification(self.add, self.remove, self.context)
        return res

    context: List[str]

    def __init__(self, **data):
        add = data.get("add")
        if isinstance(add, str):
            add = data["add"] = [add]

        remove = data.get("remove", None)
        if isinstance(remove, str):
            remove = data["remove"] = [remove]

        result = _check_specification(add, remove, data["context"])
        if len(result) == 0:
            raise ValueError(
                f"Adding: {add} and removing: {remove} resulted in no items"
            )

        super().__init__(**data)


class ParamConfig(ObsConfig):
    """
    Specification of list of model parameters. Wildcard notation is allowed.
    The syntax and meaning of  the 'add' and 'remove' keywords are the
    same as for specification of observations.

    For nodes with parameters defined by the ERT keyword GEN_KW, the
    parameter name is specified on the format: <NODENAME>:<PARAMETER_NAME>
    For nodes with parameters defined by the ERT keyword SURFACE and FIELD,
    the parameters belonging to a surface node is specified by: <NODENAME>

    Example:
    param_group:
         add: ["GEO_PARAM:*", "FIELD_PARAM*", "GENERAL:*", "SURF_PARAM_*"]
         remove: ["FIELD_PARAM_GRF1", "GEO_PARAM:A*", "GENERAL:2", "GENERAL:3"]

    (This includes all model parameters from node GEO_PARAM except parameter A,
      all field parameter nodes starting with node name FIELD_PARAM except
      FIELD_PARAM_GRF1,
      all surface parameter nodes starting with node name SURF_PARAM_)
    """

    model_config = ConfigDict(extra="forbid")


class GaussianConfig(BaseModel):
    """
    Method for calculating correlation scaling factor using a Gaussian function
    centered at the specified reference point. The ranges and orientation define
    how the scaling factor is reduced away from the reference point.

    The following equation for the scaling is defined:
    First the normalised distance is calculated:
       d = sqrt( (dx/main_range)^2 + (dy/perp_range)^2) )
    where dx is distance in azimuth direction and dy perpendicular to
    azimuth direction.

    The default scaling function is defined for gaussian decay by:
        f(d) = exp(-3 * d^2)

    This method can be used both to scale correlations for model parameter
    nodes of type FIELD and SURFACE. For nodes of type surface,
    a file specifying the grid layout of the surface must be specified.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "gaussian_decay",
        "exponential_decay",
        "const_gaussian_decay",
        "const_exponential_decay",
    ]
    main_range: Annotated[float, Field(gt=0)]
    perp_range: Annotated[float, Field(gt=0)]
    azimuth: Annotated[float, Field(ge=0.0, le=360)]
    ref_point: Annotated[List[float], Field(min_length=2, max_length=2)]
    cutoff: Optional[bool] = Field(default=False)
    surface_file: Optional[str] = None


class ExponentialConfig(GaussianConfig):
    """
    Method for calculating correlation scaling factor using Exponential function.
    See the doc string for Gaussian function for more details.
    """

    method: Literal["exponential_decay"] = "exponential_decay"


class ConstWithGaussianTaperingConfig(GaussianConfig):
    """
    Method for calculating correlation scaling factor which is 1 inside range
    and fall off using Gaussian function outside range.

    The function is defined by:
        f(d) = 1 if d <= 1
        f(d) = exp(-3 * ((d-1)/(D-1))^2 )  for  d > 1 and here D > 1.
    Here d=1 represents the ellipse defined by the range settings, and D= 1
    represents the second ellipse at which the scaling function is reduced
    to about 0.05 .

    Optionally the use of cutoff set the values for the function to 0 for d > D.
        f(d) = 0 for d > D
    This will create a discontinuity at d=D of size 0.05 for the scaling value.

    This method can be used both to scale correlations for model parameter
    nodes of type FIELD and SURFACE. For nodes of type surface,
    a file specifying the grid layout of the surface must be specified.
    """

    method: Literal["const_gaussian_decay"] = "const_gaussian_decay"
    normalised_tapering_range: Optional[Annotated[float, Field(gt=1)]] = 1.5


class ConstWithExponentialTaperingConfig(GaussianConfig):
    """
    Method for calculating correlation scaling factor which is 1 inside range
    and fall off using Exponential function outside range. See above for
    ConstWithGaussianTaperingConfig.
    """

    method: Literal["const_exponential_decay"] = "const_exponential_decay"
    normalised_tapering_range: Optional[Annotated[float, Field(gt=1)]] = 1.5


class ScalingFromFileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["from_file"]
    filename: str
    param_name: str


class ScalingForSegmentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["segment"]
    segment_filename: str
    param_name: str
    active_segments: Union[  # type: ignore
        conint(ge=0), conlist(item_type=conint(ge=0), min_length=1)
    ]
    scalingfactors: Union[  # type: ignore
        confloat(ge=0), conlist(item_type=confloat(ge=0, le=1), min_length=1)
    ]
    smooth_ranges: Optional[  # type: ignore
        conlist(item_type=conint(ge=0), min_length=2, max_length=2)
    ] = [0, 0]

    @field_validator("scalingfactors")
    @classmethod
    def check_length_consistency(cls, scalingfactors, info):
        # Ensure that active segment list and scaling factor lists are of equal length.
        active_segment_list = info.data.get("active_segments", None)
        if not active_segment_list:
            return scalingfactors
        if len(scalingfactors) != len(active_segment_list):
            raise ValueError(
                "The specified length of 'active_segments' list"
                f"{active_segment_list }\n"
                f"  and 'scalingfactors' list {scalingfactors} are different."
            )
        return scalingfactors


class ConstantScalingConfig(BaseModel):
    """
    Method for calculating constant correlation scaling factor.
    """

    model_config = ConfigDict(extra="forbid")
    method: Literal["constant"]
    value: Optional[Annotated[float, Field(strict=True, gt=0, le=1)]] = 1.0
    surface_file: Optional[str] = None


class CorrelationConfig(BaseModel):
    """
    The keyword 'correlations' specify a set of observations and model parameters
    to have active correlations. For scalar parameters coming from ERT keyword GEN_KW,
    the correlations estimated by ERT will not be reduced,
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

    model_config = ConfigDict(extra="forbid")
    name: str
    obs_group: ObsConfig
    param_group: ParamConfig
    field_scale: Optional[
        Union[
            GaussianConfig,
            ExponentialConfig,
            ConstWithGaussianTaperingConfig,
            ConstWithExponentialTaperingConfig,
            ScalingFromFileConfig,
            ScalingForSegmentsConfig,
            ConstantScalingConfig,
        ]
    ]

    surface_scale: Optional[
        Union[
            GaussianConfig,
            ExponentialConfig,
            ConstWithGaussianTaperingConfig,
            ConstWithExponentialTaperingConfig,
            ConstantScalingConfig,
        ]
    ]

    obs_context: list
    params_context: list

    def __init__(self, **data):
        data["obs_group"]["context"] = data["obs_context"]
        data["param_group"]["context"] = data["params_context"]

        if "surface_scale" in data:
            surf_scale = data["surface_scale"]
            if not isinstance(surf_scale, dict):
                raise ValueError("surface_scale must be dict")

            # String with relative path to surface files relative to config path
            if "surface_file" not in surf_scale:
                raise ValueError(
                    "Missing keyword: 'surface_file' in keyword: 'surface_scale' "
                )

            filename = pathlib.Path(surf_scale["surface_file"])
            # Check  that the file exists
            if not filename.exists():
                raise ValueError(f"File for surface: {filename} does not exist.")

            method = surf_scale.get("method")
            _valid_methods = {
                "gaussian_decay": GaussianConfig,
                "exponential_decay": ExponentialConfig,
                "const_gaussian_decay": ConstWithGaussianTaperingConfig,
                "const_exponential_decay": ConstWithExponentialTaperingConfig,
                "constant": ConstantScalingConfig,
            }

            if method not in _valid_methods:
                raise ValueError(
                    f"Unknown method: {method}, valid methods are:"
                    f" {_valid_methods.keys()}"
                )
            data["surface_scale"] = _valid_methods[method](**surf_scale)
        else:
            data["surface_scale"] = None

        if "field_scale" in data:
            field_scale = data.get("field_scale")
            # field_scale["surface_file"] = field_scale.get("surface_file", None)
            method = field_scale.get("method")
            _valid_methods = {
                "gaussian_decay": GaussianConfig,
                "exponential_decay": ExponentialConfig,
                "const_gaussian_decay": ConstWithGaussianTaperingConfig,
                "const_exponential_decay": ConstWithExponentialTaperingConfig,
                "from_file": ScalingFromFileConfig,
                "segment": ScalingForSegmentsConfig,
                "constant": ConstantScalingConfig,
            }
            if method not in _valid_methods:
                raise ValueError(
                    f"Unknown method: {method}, valid methods are:"
                    f" {_valid_methods.keys()}"
                )

            data["field_scale"] = _valid_methods[method](**field_scale)
        else:
            data["field_scale"] = None

        super().__init__(**data)


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

    model_config = ConfigDict(extra="forbid")
    observations: List[str]
    parameters: List[str]
    correlations: List[CorrelationConfig]
    log_level: Optional[Annotated[int, Field(strict=True, ge=0, le=5)]] = 1
    write_scaling_factors: Optional[bool] = False

    def __init__(self, **data):
        for correlation in data["correlations"]:
            correlation["obs_context"] = data["observations"]
            correlation["params_context"] = data["parameters"]

        super().__init__(**data)

    @field_validator("correlations", mode="after")
    @classmethod
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
