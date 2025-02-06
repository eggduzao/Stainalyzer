
"""
DAB Distribution

Usage:

# Assuming dab_dist is an instance of DABDistribution
# Assuming cluster_dist is another MultivariateLABDistribution

# Calculate the distance using the Wasserstein metric
distance = dab_dist.dab_distribution.distance_to(cluster_dist, metric='Wasserstein')

print(f"Distance (Wasserstein): {distance}")

"""

############################################################################################################
### Import
############################################################################################################

import os
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

from Stainalyzer.distribution import MultivariateLABDistribution

############################################################################################################
### Classes
############################################################################################################

class DABDistribution:
    """
    A class to manage and update a multivariate distribution for DAB-stained pixels.

    This class uses a dedicated instance of MultivariateLABDistribution to model the LAB color space
    distribution of DAB-stained pixels. It supports adding new data incrementally and recalculating
    the distribution as needed.

    Attributes:
        dab_distribution (MultivariateLABDistribution): The multivariate distribution representing
                                                        the LAB color space of DAB-stained pixels.
        total_samples (int): Total number of LAB samples used to estimate the distribution.
    """

    def __init__(self, initial_data=None, **kwargs):
        """
        Initialize the DABDistribution class.

        Parameters:
            initial_data (numpy.ndarray, optional): Initial LAB color data to create the distribution.
                                                    Shape: (n_samples, 3), where 3 corresponds to L, A, B channels.
            **kwargs: Additional parameters to be passed to MultivariateLABDistribution for initialization.
        """
        self.dab_distribution = None  # Instance of MultivariateLABDistribution
        self.total_samples = 0       # Track the total number of samples

        if initial_data is not None:
            self.add_data(initial_data, **kwargs)

    def add_data(self, lab_data, **kwargs):
        """
        Add new LAB color data to the DAB distribution.

        If the distribution does not exist, create a new one. Otherwise, update the existing distribution
        by combining the new data with the current distribution.

        Parameters:
            lab_data (numpy.ndarray): LAB color data to add. Shape: (n_samples, 3).
            **kwargs: Additional parameters to control the update process.

        Raises:
            ValueError: If lab_data is not a valid numpy array with shape (n_samples, 3).
        """

        # Validate input data
        if not isinstance(lab_data, np.ndarray) or lab_data.ndim != 2 or lab_data.shape[1] != 3:
            raise ValueError("lab_data must be a numpy array of shape (n_samples, 3).")

        # If the distribution does not exist, create a new one
        if self.dab_distribution is None:
            self.dab_distribution = MultivariateLABDistribution(lab_data, **kwargs)
            self.total_samples = lab_data.shape[0]
        else:
            # If the distribution exists, combine old and new data
            # Retrieve the existing LAB data from the current distribution
            existing_samples = self.dab_distribution.sample(self.total_samples)

            # Combine existing data with new data
            combined_data = np.vstack([existing_samples, lab_data])

            # Refit the distribution with the combined data
            self.dab_distribution.fit(combined_data, n_components=3, **kwargs)

            # Update the total number of samples
            self.total_samples += lab_data.shape[0]

    def evaluate(self, lab_data):
        """
        Evaluate the likelihood of given LAB data against the DAB distribution.

        Parameters:
            lab_data (numpy.ndarray): LAB color data to evaluate. Shape: (n_samples, 3).

        Returns:
            numpy.ndarray: Probability density values for the input LAB data. Shape: (n_samples,).

        Raises:
            ValueError: If the DAB distribution has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("The DAB distribution is not fitted. Please fit the distribution before evaluating.")

        # Evaluate the likelihood using the underlying MultivariateLABDistribution
        return self.dab_distribution.evaluate(lab_data)

    def sample(self, n_samples=100):
        """
        Generate random samples from the DAB distribution.

        Parameters:
            n_samples (int): Number of samples to generate.

        Returns:
            numpy.ndarray: LAB samples drawn from the DAB distribution. Shape: (n_samples, 3).

        Raises:
            ValueError: If the DAB distribution is not fitted or initialized.
        """
        if self.dab_distribution is None:
            raise ValueError("The DAB distribution is not fitted. Please add data before sampling.")

        # Generate samples from the underlying MultivariateLABDistribution
        return self.dab_distribution.sample(n_samples)

    def save(self, file_path):
        """
        Save the DAB distribution to a file.

        This method serializes the instance of DABDistribution, including the underlying
        MultivariateLABDistribution and the total number of samples.

        Parameters:
            file_path (str): Path to save the distribution object.

        Raises:
            ValueError: If the file_path is invalid or saving fails.
        """
        try:
            # Serialize the DABDistribution object using pickle
            with open(file_path, 'wb') as file:
                pickle.dump(self, file)
        except Exception as e:
            raise ValueError(f"Failed to save the DAB distribution to {file_path}. Original error: {e}")

    @property
    def is_fitted(self):
        """
        Check if the DAB distribution has been fitted.

        This property verifies whether the underlying MultivariateLABDistribution
        exists and has been fitted with data.

        Returns:
            bool: True if the distribution exists and is fitted, False otherwise.
        """
        return self.dab_distribution is not None and self.dab_distribution.model is not None

    @staticmethod
    def load(file_path):
        """
        Load a DAB distribution from a file.

        This method deserializes a saved instance of DABDistribution, including
        the underlying MultivariateLABDistribution and the total sample count.

        Parameters:
            file_path (str): Path to load the distribution object from.

        Returns:
            DABDistribution: Loaded DAB distribution object.

        Raises:
            ValueError: If the file_path is invalid or loading fails.
        """
        try:
            with open(file_path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            raise ValueError(f"Failed to load the DAB distribution from {file_path}. Original error: {e}")

    def __repr__(self):
        """
        Representation of the DABDistribution class.

        Provides a summary of the DAB distribution's current state, including whether
        it is fitted and the total number of samples.

        Returns:
            str: Summary of the current state of the DABDistribution object.
        """
        if self.is_fitted:
            num_components = len(self.dab_distribution.weights_)
            return (f"DABDistribution (Fitted: Yes, Components: {num_components}, "
                    f"Total Samples: {self.total_samples})")
        else:
            return "DABDistribution (Fitted: No, Total Samples: 0)"

