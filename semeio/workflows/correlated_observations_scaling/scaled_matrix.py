import logging
from copy import deepcopy

import numpy as np

from semeio.workflows.correlated_observations_scaling.exceptions import (
    EmptyDatasetException,
)


class DataMatrix:
    def __init__(self, input_data):
        """
        Takes input data in the form of a Pandas multi index dataframe with
        observations, standard deviation and simulated values. Assumes observations
        are prepended with _OBS and standard deviation with _STD.
        """
        self.data = input_data
        if input_data.shape[1] == 0:
            raise EmptyDatasetException("Empty dataset")

    def get_data_matrix(self):
        """
        Extracts data from a dataframe and returns a matrix
        """
        return self.data[~self.data.index.isin(["OBS", "STD"])].values

    def normalize_by_std(self):
        """
        Normalize the matrix in place.
        """
        self.data = self.get_normalized_by_std()

    def get_normalized_by_std(self):
        """
        Duplicates the behavior of obs_data_scale, and scales the simulated data
        by 1 / (observation standard deviation), per observation key, i.e. each
        simulation data point is scaled by its corresponding std deviation
        from observations and returns a copy.
        """
        output_data = deepcopy(self.data)
        data_matrix = self._get_data()
        std_vector = self.data.loc["STD"]
        output_data[~output_data.index.isin(["OBS", "STD"])] = data_matrix * (
            1.0 / std_vector
        )

        return output_data

    @staticmethod
    def get_scaling_factor(nr_observations, nr_components):
        """
        Calculates a observation scaling factor which is:
            sqrt(nr_obs / pc)
        where:
            nr_obs is the number of observations
            pc is the number of primary components from PCA analysis
                below a user threshold
        """
        logging.info(  # pylint: disable=logging-fstring-interpolation
            (
                f"Calculation scaling factor, nr of primary components: "
                f"{nr_components}, number of observations: {nr_observations}"
            )
        )
        return np.sqrt(nr_observations / float(nr_components))

    def _get_data(self):
        return self.data[~self.data.index.isin(["OBS", "STD"])]

    def get_nr_primary_components(self, threshold):
        """
        Takes a matrix, does PCA and calculates the cumulative variance ratio
        and returns an int which is the number of primary components where
        the cumulative variance is smaller than user set threshold. Notice the
        way of calculating number of primary components. This is done to
        replicate existing behavior:

        int num_significant  = 0;
        {
        double running_sigma2  = 0;
        for (int i=0; i < num_singular_values; i++) {
          if (running_sigma2 / total_sigma2 < truncation) {
             num_significant++;
             running_sigma2 += sig0[i] * sig0[i];
          } else
             break;
        }

        Also returns an array of singular values.
        """
        data_matrix = self.get_data_matrix()
        data_matrix = data_matrix - data_matrix.mean(axis=0)
        _, singulars, _ = np.linalg.svd(data_matrix.astype(float), full_matrices=False)
        variance_ratio = np.cumsum(singulars**2) / np.sum(singulars**2)
        return len([1 for i in variance_ratio[:-1] if i < threshold]) + 1, singulars
