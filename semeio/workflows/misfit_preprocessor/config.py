import collections
import configsuite as cs
import copy
import fnmatch


_SCALING = "scaling"
_THRESHOLD = "threshold"
_STD_CUTOFF = "std_cutoff"
_ALPHA = "alpha"
_CLUSTERING = "clustering"
_METHOD = "method"
SPEARMAN_CORRELATION = "spearman_correlation"
AUTO_SCALE = "auto_scale"
_FCLUSTER = "fcluster"
_SPEARMAN_THRESHOLD = "t"
_CRITERION = "criterion"
_INCONSISTENT = "inconsistent"
_DISTANCE = "distance"
_MAXCLUST = "maxclust"
_MONOCRIT = "monocrit"
_MAXCLUST_MONOCRIT = "maxclust_monocrit"
_DEPTH = "depth"
_LINKAGE = "linkage"
_SINGLE = "single"
_AVERAGE = "average"
_METHODS = (
    _SINGLE,
    "complete",
    _AVERAGE,
    "weighted",
    "centroid",
    "ward",
)
_METRIC = "metric"
_EUCLIDEAN = "euclidean"
_METRICS = (
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    _EUCLIDEAN,
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulsinski",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
)
_OBSERVATIONS = "observations"


_DEFAULT_VALUES = {
    _OBSERVATIONS: (),
}


def _bounds_validator(
    lower=None, lower_inclusive=True, upper=None, upper_inclusive=True
):
    msg = "Value must be within the interval {}, {}".format(
        "<"
        if lower is None
        else "{}{}".format(
            "[" if lower_inclusive else "(",
            lower,
        ),
        ">"
        if upper is None
        else "{}{}".format(
            upper,
            "]" if upper_inclusive else ")",
        ),
    )

    @cs.validator_msg(msg)
    def validator(value):
        lower_valid = lower is None or (
            lower <= value if lower_inclusive else lower < value
        )
        upper_valid = upper is None or (
            value <= upper if upper_inclusive else value < upper
        )
        return lower_valid and upper_valid

    return validator


def _one_of(*valid_values):
    msg = "Value must be one of: ({})".format(", ".join(map(str, valid_values)))

    @cs.validator_msg(msg)
    def validator(value):
        return value in valid_values

    return validator


@cs.validator_msg("t must be an integer if a max cluster criteria is choosen")
def _t_is_int_if_maxclust(spearman_config):
    if spearman_config[_CRITERION] in (_MAXCLUST, _MAXCLUST_MONOCRIT):
        return isinstance(spearman_config[_SPEARMAN_THRESHOLD], int)
    return True


@cs.transformation_msg(
    (
        "Will be defaulted to 5 for all max cluster criterias and "
        "1.15 for all other criterias"
    )
)
def _inject_default_t(config):
    config = copy.deepcopy(config)
    criterion = config.get(_CRITERION)
    if _SPEARMAN_THRESHOLD in config:
        pass
    elif criterion in (_MAXCLUST, _MAXCLUST_MONOCRIT):
        config[_SPEARMAN_THRESHOLD] = 5
    elif criterion in (_INCONSISTENT, _DISTANCE, _MONOCRIT):
        config[_SPEARMAN_THRESHOLD] = 1.15
    else:
        raise ValueError("Unknown criterion: {}".format(criterion))
    return config


_LINKAGE_SCHEMA = {
    cs.MetaKeys.Type: cs.types.NamedDict,
    cs.MetaKeys.Description: (
        "The {linkage} implementation is backed by scipy and for a "
        "more detailed description we refer the reader to the "
        "documentation of scipy.cluster.hierarchy.linkage` "
        "(https://docs.scipy.org/doc/scipy/reference/generated"
        "/scipy.cluster.hierarchy.linkage.html)."
    ).format(linkage=_LINKAGE),
    cs.MetaKeys.Content: {
        _METHOD: {
            cs.MetaKeys.Type: cs.types.String,
            cs.MetaKeys.Description: (
                "Method used to calculate the distance between the clusters."
            ),
            cs.MetaKeys.ElementValidators: (_one_of(*_METHODS),),
            cs.MetaKeys.Default: _AVERAGE,
        },
        _METRIC: {
            cs.MetaKeys.Type: cs.types.String,
            cs.MetaKeys.Description: ("Distance metric used to calculate distances."),
            cs.MetaKeys.ElementValidators: (_one_of(*_METRICS),),
            cs.MetaKeys.Default: _EUCLIDEAN,
        },
    },
}


_FCLUSTER_SCHEMA = {
    cs.MetaKeys.Type: cs.types.NamedDict,
    cs.MetaKeys.Description: (
        "The {fcluster} implementation is backed by scipy and for a "
        "more detailed description we refer the reader to the "
        "documentation of scipy.cluster.hierarchy.fcluster` "
        "(https://docs.scipy.org/doc/scipy/reference/generated"
        "/scipy.cluster.hierarchy.fcluster.html)."
    ).format(fcluster=_FCLUSTER),
    cs.MetaKeys.ElementValidators: (_t_is_int_if_maxclust,),
    cs.MetaKeys.Transformation: _inject_default_t,
    cs.MetaKeys.Content: {
        _SPEARMAN_THRESHOLD: {
            cs.MetaKeys.Type: cs.types.Number,
            cs.MetaKeys.Description: (
                "Scalar threshold for the clustering. When a "
                "'{maxclust}' {criterion} is used, the scalar gives "
                "the maximum number of clusters to be formed."
            ).format(maxclust=_MAXCLUST, criterion=_CRITERION),
            cs.MetaKeys.ElementValidators: (
                _bounds_validator(lower=0, lower_inclusive=False),
            ),
        },
        _CRITERION: {
            cs.MetaKeys.Type: cs.types.String,
            cs.MetaKeys.Description: (
                "The criterion to use in forming flat clusters. "
                "Defaults to {default_criterion}."
            ).format(default_criterion=_INCONSISTENT),
            cs.MetaKeys.ElementValidators: (
                _one_of(
                    _INCONSISTENT,
                    _DISTANCE,
                    _MAXCLUST,
                    _MONOCRIT,
                    _MAXCLUST_MONOCRIT,
                ),
            ),
            cs.MetaKeys.Default: _INCONSISTENT,
        },
        _DEPTH: {
            cs.MetaKeys.Type: cs.types.Integer,
            cs.MetaKeys.Description: (
                "The maximum depth to perform the {inconsistent} calculation. "
            ).format(
                inconsistent=_INCONSISTENT,
            ),
            cs.MetaKeys.ElementValidators: (_bounds_validator(lower=1),),
            cs.MetaKeys.Default: 2,
        },
    },
}


AUTO_SCALE_SCHEMA = {
    cs.MetaKeys.Type: cs.types.NamedDict,
    cs.MetaKeys.Description: (
        "The {sc} clustering is supporting multiple parameters."
    ).format(sc=AUTO_SCALE),
    cs.MetaKeys.Content: {_LINKAGE: _LINKAGE_SCHEMA},
}


SPEARMAN_CORRELATION_SCHEMA = {
    cs.MetaKeys.Type: cs.types.NamedDict,
    cs.MetaKeys.Description: (
        "The {sc} clustering is supporting multiple parameters."
    ).format(sc=SPEARMAN_CORRELATION),
    cs.MetaKeys.Content: {_FCLUSTER: _FCLUSTER_SCHEMA, _LINKAGE: _LINKAGE_SCHEMA},
}


_CLUSTERING_SCHEMA = {
    cs.MetaKeys.Type: cs.types.NamedDict,
    cs.MetaKeys.Description: (
        "The clustering supports tweaking multiple parameters if the initial "
        "results are not according to expectations."
    ),
    cs.MetaKeys.Content: {
        _METHOD: {
            cs.MetaKeys.Type: cs.types.String,
            cs.MetaKeys.Description: (
                "Currently the workflow only supports clustering by "
                "{spearman}.".format(spearman=SPEARMAN_CORRELATION)
            ),
            cs.MetaKeys.ElementValidators: (_one_of(SPEARMAN_CORRELATION, AUTO_SCALE),),
            cs.MetaKeys.Default: AUTO_SCALE,
        },
        SPEARMAN_CORRELATION: SPEARMAN_CORRELATION_SCHEMA,
        AUTO_SCALE: AUTO_SCALE_SCHEMA,
    },
}


_SCALING_SCHEMA = {
    cs.MetaKeys.Type: cs.types.NamedDict,
    cs.MetaKeys.Description: ("Computes primary components and scales them."),
    cs.MetaKeys.Content: {
        _THRESHOLD: {
            cs.MetaKeys.Type: cs.types.Number,
            cs.MetaKeys.Description: (
                "Threshold used when computing primary components of the "
            ),
            cs.MetaKeys.Default: 0.95,
            cs.MetaKeys.ElementValidators: (
                _bounds_validator(
                    lower=0,
                    lower_inclusive=False,
                    upper=1,
                    upper_inclusive=False,
                ),
            ),
        },
        _STD_CUTOFF: {
            cs.MetaKeys.Type: cs.types.Number,
            cs.MetaKeys.Description: (
                "A lower bound on the ensemble standard deviation. All data "
                "points with insufficient variation will be dropped."
            ),
            cs.MetaKeys.AllowNone: True,
            cs.MetaKeys.Required: False,
        },
        _ALPHA: {
            cs.MetaKeys.Type: cs.types.Number,
            cs.MetaKeys.Description: (
                "Scalar controlling the allowed distance between "
                "ensemble mean and observation. In particular, if: "
                "`abs(observed_value - ensemble_mean) "
                "> alpha * (ensenmble_std + observed_std)` "
                "the data point will be dropped."
            ),
            cs.MetaKeys.AllowNone: True,
            cs.MetaKeys.Required: False,
        },
    },
}


class _BooleanWithMessage:
    def __init__(self, value, msg):
        self._value = value
        self._msg = msg

    def __nonzero__(self):
        return self._value is True

    def __bool__(self):
        return self._value is True

    def __and__(self, other):
        return bool(self) and bool(other)

    @property
    def msg(self):
        return self._msg


# NB: This is a validator, not a transformation.
# This is a hack to avoid configsuite wrapping the return value
@cs.transformation_msg("Ensures that all requested observations are indeed present")
def _observations_present(observation_key, context):
    if observation_key in context.observation_keys:
        return _BooleanWithMessage(
            True,
            "Observation {} found".format(observation_key),
        )
    else:
        return _BooleanWithMessage(
            False,
            "Found no match for observation {}".format(observation_key),
        )


@cs.transformation_msg("Inject all as default and expand filters")
def _realize_filters(observation_keys, context):
    all_keys = tuple(context.observation_keys)
    if len(observation_keys) == 0:
        observation_keys = ("*",)

    matches = set()
    for obs_filter in observation_keys:
        new_matches = set(fnmatch.filter(all_keys, obs_filter))
        if len(new_matches) == 0:
            new_matches = set((obs_filter,))
        matches = matches.union(new_matches)

    return tuple(matches)


_OBSERVATION_SCHEMA = {
    cs.MetaKeys.Type: cs.types.List,
    cs.MetaKeys.Description: (
        "By default all observations are clustered. If this is not desired "
        "one can provide a list of observation names to cluster. Wildcards are "
        'supported. Example: ["OP_1_WWCT*", "SHUT_IN_OP1"]'
    ),
    cs.MetaKeys.ContextTransformation: _realize_filters,
    cs.MetaKeys.Content: {
        cs.MetaKeys.Item: {
            cs.MetaKeys.Type: cs.types.String,
            cs.MetaKeys.ContextValidators: (_observations_present,),
        }
    },
}


_SCHEMA = {
    cs.MetaKeys.Type: cs.types.NamedDict,
    cs.MetaKeys.Description: (
        "The Misfit Preprocessor workflow provides the users with "
        "tooling to cluster and scale observations to prevent overfitting to "
        "all or parts of the observations. Examples where overfitting is likely "
        "to happen are accumulative time series and when one have data sources "
        "with severely different sampling magnitudes. This workflow is intended "
        "to be an almost out-of-the-box solution, where the user can tweak some "
        "parameters to get reasonable clusterings. The rest will be configured "
        "according to whatever is at the time considered best practices."
    ),
    cs.MetaKeys.Content: {
        _OBSERVATIONS: _OBSERVATION_SCHEMA,
        _CLUSTERING: _CLUSTERING_SCHEMA,
        _SCALING: _SCALING_SCHEMA,
    },
}


class _ObservationContext:  # pylint: disable=too-few-public-methods
    def __init__(self, observation_keys):
        Context = collections.namedtuple("Context", ("observation_keys",))
        self._context = Context(set(observation_keys))

    def __call__(self, _):
        return self._context


class MisfitPreprocessorConfig(cs.ConfigSuite):
    def __init__(self, config_data, observation_keys):
        super().__init__(
            config_data,
            _SCHEMA,
            (_DEFAULT_VALUES,),
            extract_validation_context=_ObservationContext(observation_keys),
            extract_transformation_context=_ObservationContext(observation_keys),
            deduce_required=True,
        )


def assemble_config(misfit_preprocessor_config, measured_data):
    observation_names = set(col[0] for col in measured_data.data.columns)
    return MisfitPreprocessorConfig(misfit_preprocessor_config, observation_names)
