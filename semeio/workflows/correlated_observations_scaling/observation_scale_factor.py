from semeio.workflows.correlated_observations_scaling.scaled_matrix import DataMatrix


class ObservationScaleFactor:
    def __init__(
        self,
        reporter,
        measured_data,
    ):
        """Creates a ScalingJob instance with the given obs_keys, obs,
        obs_with_data and user_config_dict."""
        self._reporter = reporter
        self._measured_data = measured_data

    def perform_pca(self, threshold):
        matrix = DataMatrix(self._measured_data.data)
        matrix.normalize_by_std()

        nr_components, singular_values = matrix.get_nr_primary_components(
            threshold=threshold
        )
        self._reporter.publish("svd", list(singular_values))
        return nr_components, singular_values

    def get_scaling_factor(self, threshold):
        """
        Collects data performs pca, and returns scaling factor, assumes validated input.
        """
        nr_observations = self._measured_data.data.shape[1]
        nr_components, _ = self.perform_pca(threshold)
        scale_factor = DataMatrix.get_scaling_factor(nr_observations, nr_components)

        self._reporter.publish("scale_factor", scale_factor)

        return scale_factor
