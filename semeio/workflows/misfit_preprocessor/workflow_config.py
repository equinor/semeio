from typing import Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from pydantic import confloat, validator, BaseModel, Field

from semeio.workflows.misfit_preprocessor.hierarchical_config import (
    HierarchicalConfig,
    LimitedHierarchicalConfig,
    BaseMisfitPreprocessorConfig,
)
from semeio.workflows.misfit_preprocessor.kmeans_config import (
    KmeansClustering,
    LimitedKmeansClustering,
)


#  pylint: disable=too-few-public-methods,no-self-argument


class PCAConfig(BaseMisfitPreprocessorConfig):
    threshold: confloat(gt=0, le=1.0) = 0.95


class AutoScaleConfig(BaseMisfitPreprocessorConfig):
    type: Literal["auto_scale"] = "auto_scale"
    clustering: Union[
        LimitedHierarchicalConfig, LimitedKmeansClustering
    ] = LimitedHierarchicalConfig()
    pca: PCAConfig = PCAConfig()


class CustomScaleConfig(BaseMisfitPreprocessorConfig):
    type: Literal["custom_scale"] = "custom_scale"
    clustering: Union[HierarchicalConfig, KmeansClustering] = HierarchicalConfig()
    pca: PCAConfig = PCAConfig()


class MisfitConfig(BaseMisfitPreprocessorConfig):
    """
    The Misfit Preprocessor workflow provides the users with
    tooling to cluster and scale observations to prevent overfitting to
    all or parts of the observations. Examples where overfitting is likely
    to happen are accumulative time series and when one have data sources
    with severely different sampling magnitudes. This workflow is intended
    to be an almost out-of-the-box solution, where the user can tweak some
    parameters to get reasonable clusterings. The rest will be configured
    according to whatever is at the time considered best practices.
    The recommended practice for running this workflow is though the
    ert workflow hooks. The hook needs to run after the simulations, as
    simulated data is needed for the workflow. The relevant hooks are
    POST_SIMULATION, PRE_FIRST_UPDATE and PRE_UPDATE. The recommended
    hook is PRE_FIRST_UPDATE, which will only run once, also in the case
    where an iterative algorithm is used.
    """

    workflow: Union[AutoScaleConfig, CustomScaleConfig] = Field(AutoScaleConfig())
    # "By default all observations are clustered. If this is not desired "
    # "one can provide a list of observation names to cluster. Wildcards are "
    # 'supported. Example: ["OP_1_WWCT*", "SHUT_IN_OP1"]'
    observations: List[str] = []

    @validator("workflow", pre=True)
    def validate_workflow(cls, value):
        """
        To improve the user feedback we explicitly check if the type of workflow
        is configured, and if it is, we bypass the Union. If it has not been given
        we let the Union do the validation.
        """
        if isinstance(value, BaseModel):
            return value
        if not isinstance(value, dict):
            raise ValueError("value must be dict")
        if "type" not in value:
            return value
        workflow = value.get("type")
        if workflow == "auto_scale":
            return AutoScaleConfig(**value)
        elif workflow == "custom_scale":
            return CustomScaleConfig(**value)
        else:
            raise ValueError(f"Unknown workflow {workflow}")
