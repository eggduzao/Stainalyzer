
"""
distribution
------------

"""

############################################################################################################
### Import
############################################################################################################

import os
import numpy as np

############################################################################################################
### Constants
############################################################################################################

# Constants
SEED = 1987
np.random.seed(SEED)

############################################################################################################
### Multivariate Gaussian Class
############################################################################################################

class GaussianDistribution:
    """
    A class to represent a Gaussian (Normal) distribution in n-dimensional space.

    Attributes:
    -----------
    mean : np.ndarray
        The mean vector of the Gaussian distribution.
    covariance : np.ndarray
        The covariance matrix of the Gaussian distribution.
    n_samples : int
        The number of samples of the Gaussian distribution.

    Methods:
    --------
    get_mean():
        Returns the mean vector.
    
    set_mean(mean):
        Sets the mean vector.
    
    get_covariance():
        Returns the covariance matrix.
    
    set_covariance(covariance):
        Sets the covariance matrix.

    def get_n_samples(self):
        Returns the n_samples value of the Gaussian distribution.

    def set_n_samples(self, n_samples):
        Sets the covariance value of the Gaussian distribution.

    def get_mean_covariance_and_samples(self):
        Returns the mean vector, covariance matrix and number of samples of the Gaussian distribution.

    def set_mean_covariance_and_samples(self, mean, covariance, n_samples):
        Sets the mean vector, the covariance matrix and number of samples of the Gaussian distribution.

    __repr__():
        Returns a detailed string representation of the GaussianDistribution object.

    __str__():
        Returns a user-friendly string representation of the GaussianDistribution object.
    """
    def __init__(self, mean=None, covariance=None, n_samples=None):
        """
        Initializes a GaussianDistribution object with mean and covariance.

        Parameters:
        -----------
        mean : array-like, optional
            The mean vector of the Gaussian distribution. Defaults to None.
        covariance : array-like, optional
            The covariance matrix of the Gaussian distribution. Defaults to None.
        n_samples : int, optional
            The number of samples of the Gaussian distribution. Defaults to None.
        """
        self.mean = np.array(mean) if mean is not None else None
        self.covariance = np.array(covariance) if covariance is not None else None
        self.n_samples = np.int32(n_samples) if n_samples is not None else 0

    def get_mean(self):
        """
        Returns the mean vector of the Gaussian distribution.

        Returns:
        --------
        np.ndarray
            The mean vector.
        """
        return self.mean

    def set_mean(self, mean):
        """
        Sets the mean vector of the Gaussian distribution.

        Parameters:
        -----------
        mean : array-like
            The mean vector to set.
        """
        self.mean = np.array(mean)

    def get_covariance(self):
        """
        Returns the covariance matrix of the Gaussian distribution.

        Returns:
        --------
        np.ndarray
            The covariance matrix.
        """
        return self.covariance

    def set_covariance(self, covariance):
        """
        Sets the covariance matrix of the Gaussian distribution.

        Parameters:
        -----------
        covariance : array-like
            The covariance matrix to set.
        """
        self.covariance = np.array(covariance)

    def get_n_samples(self):
        """
        Returns the n_samples value of the Gaussian distribution.

        Returns:
        --------
        np.ndarray
            The n_samples value.
        """
        return self.n_samples

    def set_n_samples(self, n_samples):
        """
        Sets the covariance value of the Gaussian distribution.

        Parameters:
        -----------
        covariance : array-like
            The covariance value to set.
        """
        self.n_samples = np.array(n_samples)

    def get_mean_covariance_and_samples(self):
        """
        Returns the mean vector, covariance matrix and number of samples of the Gaussian distribution.

        Returns:
        --------
        tuple
            A tuple containing the mean vector and the covariance matrix.
        """
        return self.mean, self.covariance, self.n_samples

    def set_mean_covariance_and_samples(self, mean, covariance, n_samples):
        """
        Sets the mean vector, the covariance matrix and number of samples of the Gaussian distribution.

        Parameters:
        -----------
        mean : array-like
            The mean vector to set.
        covariance : array-like
            The covariance matrix to set.
        n_samples : array-like
            The n_samples value to set.
        """
        self.mean = np.array(mean)
        self.covariance = np.array(covariance)
        self.n_samples = np.array(n_samples)

    def __repr__(self):
        """
        Returns a detailed string representation of the GaussianDistribution object.

        Returns:
        --------
        str
            A string representation of the object.
        """
        return f"GaussianDistribution(mean={self.get_mean()}, covariance={self.get_covariance()})"

    def __str__(self):
        """
        Returns a user-friendly string representation of the GaussianDistribution object.

        Returns:
        --------
        str
            A string describing the Gaussian distribution.
        """
        return (
            f"Gaussian Distribution\n"
            f"Mean: {self.mean}\n"
            f"Covariance Matrix: {self.covariance}"
        )

"""
class MultivariateLABDistribution:
    
    A class to store and manage multivariate LAB color distributions.

    This class acts as a container for Gaussian Multivariate LAB Distributions or
    other distribution estimation methods, supporting operations such as fitting,
    sampling, evaluation, and comparison using various metrics.

    Attributes:
        model (object): The fitted distribution object (e.g., Gaussian Mixture Model or Kernel Density Estimator).
        mean_ (numpy.ndarray): Mean of the fitted distribution. Shape: (n_components, 3).
        covariance_ (numpy.ndarray): Covariance matrix of the fitted distribution. Shape: (n_components, 3, 3).
        weights_ (numpy.ndarray): Weights of the components in the distribution (if applicable). Shape: (n_components,).
    

    def __init__(self, lab_data=None, method='GMM', **kwargs):
        "" "
        Initialize the MultivariateLABDistribution class.

        Parameters:
            lab_data (numpy.ndarray, optional): LAB color data used to initialize and fit the distribution.
                                                Shape: (n_samples, 3), where 3 corresponds to L, A, B channels.
            method (str, optional): Method for estimating the distribution. Currently supports:
                                    - 'GMM' (Gaussian Mixture Model, default)
                                    - 'KDE' (Kernel Density Estimator, in future extensions).
            **kwargs: Additional parameters for the fitting method, such as `n_components` for GMM.

        Attributes Initialized:
            model (object): The fitted model, initialized as None until `fit` is called.
            mean_ (numpy.ndarray): Mean of the fitted distribution, initialized as None.
            covariance_ (numpy.ndarray): Covariance matrix of the fitted distribution, initialized as None.
            weights_ (numpy.ndarray): Weights of the components in the distribution, initialized as None.

        Raises:
            ValueError: If the provided `method` is not supported.

        Example Usage:
            >>> lab_data = np.random.rand(100, 3) * 100  # Example LAB data
            >>> dist = MultivariateLABDistribution(lab_data=lab_data, method='GMM', n_components=3)
            >>> dist.fit(lab_data)
        "" "
        # Initialize attributes
        self.model = None
        self.mean_ = None
        self.covariance_ = None
        self.weights_ = None

        # Automatically fit the distribution if lab_data is provided
        if lab_data is not None:
            self.fit(lab_data, method=method, **kwargs)

        # Plotting attributes
        self.plotting_utils = PlottingUtils()

    def fit(self, lab_data, method="GMM", n_components=1, reg_covar=1e-6, **kwargs):
        "" "
        Fit a multivariate distribution to the given LAB data.

        Parameters:
            lab_data (numpy.ndarray): LAB color data to fit the distribution.
                                      Shape: (n_samples, 3) where 3 corresponds to L, A, B channels.
            method (str): The method to use for fitting. Currently, only "GMM" (Gaussian Mixture Model) is supported.
                          Default is "GMM".
            **kwargs: Additional parameters for the fitting method (e.g., `n_components` for GMM).

        Raises:
            ValueError: If the method is not supported or the input data is invalid.
            ValueError: If lab_data has fewer data than n_components, GMM fitting can fail.
            ValueError: If method is not 'GMM'.
        "" "

        # Validate input data
        if not isinstance(lab_data, np.ndarray) or lab_data.ndim != 2 or lab_data.shape[1] != 3:
            raise ValueError("lab_data must be a numpy array of shape (n_samples, 3).")

        if lab_data.shape[0] < n_components:
            raise ValueError(f"Insufficient data: {lab_data.shape[0]} samples for {n_components} components.")

        if method.upper() != "GMM":
            raise ValueError(f"Unsupported fitting method: {method}. Supported methods are: ['GMM'].")

        # Handle method
        if method.upper() == "GMM":
            # Number of components (default to 1 if not specified)
            n_components = kwargs.get("n_components", 1)

            # Initialize and fit the Gaussian Mixture Model
            gmm = GaussianMixture(
                                  n_components=n_components, 
                                  covariance_type="full", 
                                  reg_covar=reg_covar,  # Adds small value to diagonal of covariance matrices
                                  **kwargs
                                  )
            gmm.fit(lab_data)

            # Store the fitted model and important parameters
            self.model = gmm
            self.mean_ = gmm.means_
            self.covariance_ = gmm.covariances_
            self.weights_ = gmm.weights_

            # Assert shapes to validate GMM output
            assert self.mean_.shape == (n_components, 3), f"Unexpected mean shape: {self.mean_.shape}"
            assert self.covariance_.shape == (n_components, 3, 3), f"Unexpected covariance shape: {self.covariance_.shape}"
            assert self.weights_.shape == (n_components,), f"Unexpected weights shape: {self.weights_.shape}"
        else:
            raise ValueError(f"Unsupported fitting method: {method}. Only 'GMM' is currently supported.")

    def __repr__(self):
        "" "
        Representation of the class with the fitted model's details.
        "" "
        if self.model is None:
            return "MultivariateLABDistribution (Unfitted)"
        else:
            return f"MultivariateLABDistribution (Fitted with {len(self.weights_)} components)"

    def sample(self, n_samples=100):
        "" "
        Generate random samples from the fitted distribution.

        Parameters:
            n_samples (int): Number of samples to generate.

        Returns:
            numpy.ndarray: Array of LAB samples drawn from the distribution.
                           Shape: (n_samples, 3) where 3 corresponds to L, A, B channels.

        Raises:
            ValueError: If the model is not fitted before calling this method.
        "" "
        if self.model is None:
            raise ValueError("The model is not fitted. Please fit the distribution before sampling.")

        # Generate samples from the fitted GMM
        samples, _ = self.model.sample(n_samples)

        return samples

    def evaluate(self, lab_data):
        "" "
        Evaluate the probability density or likelihood of given LAB data.

        Parameters:
            lab_data (numpy.ndarray): LAB color data to evaluate.
                                      Shape: (n_samples, 3), where each row corresponds to L, A, B values.

        Returns:
            numpy.ndarray: Array of probability density values for each input LAB data point.
                           Shape: (n_samples,).

        Raises:
            ValueError: If the model is not fitted before calling this method.
        "" "
        if self.model is None:
            raise ValueError("The model is not fitted. Please fit the distribution before evaluating.")

        # Evaluate the probability density of the input LAB data using the fitted GMM
        density = self.model.score_samples(lab_data)

        # Convert log-likelihood to probability density
        return np.exp(density)

    def distance_to(self, other_distribution, metric='Wasserstein', weights=None):
        "" "
        Calculate the distance between this distribution and another LAB distribution.

        This method supports multivariate Gaussian Mixture Models (GMMs) with an arbitrary 
        number of components. It dynamically adjusts to the number of components and computes 
        the specified distance metric between the two distributions.

        Parameters:
            other_distribution (MultivariateLABDistribution): Another LAB distribution to compare.
            metric (str): Distance metric to use. Supported metrics:
                          - 'Wasserstein': 2-Wasserstein distance between two distributions.
                          - 'KL-divergence': Kullback-Leibler divergence.
                          - 'Bhattacharyya': Bhattacharyya distance.
                          Default is 'Wasserstein'.
            weights (tuple or list, optional): Tuple of weights (wl, wa, wb) for the L, A, B channels.
                                               If None, no weighting is applied. Default is None.

        Returns:
            float: The calculated distance between the two distributions.

        Raises:
            ValueError: If the provided metric is not supported or the models are not fitted.
        "" "
        if self.model is None or other_distribution.model is None:
            raise ValueError("Both distributions must be fitted before calculating the distance.")

        if weights is not None:
            if len(weights) != 3:
                raise ValueError("Weights must be a tuple or list with exactly three elements: (wl, wa, wb).")
            weights = np.array(weights)
        else:
            weights = np.ones(3)  # Default weights are equal for L, A, B channels

        # Determine the number of components in each distribution
        n_components_self = len(self.mean_)
        n_components_other = len(other_distribution.mean_)

        # Initialize overall distance
        total_distance = 0.0

        # Loop over all component pairs (self vs. other)
        for i in range(n_components_self):
            mean1, cov1, weight1 = self.mean_[i], self.covariance_[i], self.weights_[i]
            for j in range(n_components_other):
                mean2, cov2, weight2 = other_distribution.mean_[j], other_distribution.covariance_[j], other_distribution.weights_[j]

                # Apply weights to means and covariances
                mean1_w = mean1 * weights
                mean2_w = mean2 * weights
                cov1_w = cov1 * weights[:, None] * weights[None, :]
                cov2_w = cov2 * weights[:, None] * weights[None, :]

                if metric.lower() == 'wasserstein':
                    # Calculate the 2-Wasserstein distance
                    mean_diff = np.linalg.norm(mean1_w - mean2_w)
                    try:
                        cov_diff = np.trace(cov1_w + cov2_w - 2 * linalg.sqrtm(cov1_w @ cov2_w))
                    except ValueError as e:
                        raise ValueError(f"Covariance matrix calculation failed: {e}")
                    component_distance = mean_diff + cov_diff

                elif metric.lower() == 'kl-divergence':
                    # KL Divergence between two multivariate Gaussian distributions
                    det_cov1 = np.linalg.det(cov1_w)
                    det_cov2 = np.linalg.det(cov2_w)
                    inv_cov2 = np.linalg.inv(cov2_w)

                    term1 = np.trace(inv_cov2 @ cov1_w)
                    term2 = (mean2_w - mean1_w).T @ inv_cov2 @ (mean2_w - mean1_w)
                    term3 = np.log(det_cov2 / det_cov1)
                    component_distance = 0.5 * (term1 + term2 - 3 + term3)

                elif metric.lower() == 'bhattacharyya':
                    # Bhattacharyya distance between two multivariate Gaussian distributions
                    cov_avg = (cov1_w + cov2_w) / 2
                    mean_diff = mean1_w - mean2_w

                    term1 = 0.125 * mean_diff.T @ np.linalg.inv(cov_avg) @ mean_diff
                    term2 = 0.5 * np.log(np.linalg.det(cov_avg) / np.sqrt(np.linalg.det(cov1_w) * np.linalg.det(cov2_w)))
                    component_distance = term1 + term2

                else:
                    raise ValueError(f"Unsupported distance metric: {metric}. Supported metrics are 'Wasserstein', 'KL-divergence', and 'Bhattacharyya'.")

                # Weight the component distance by the product of component weights
                total_distance += weight1 * weight2 * component_distance

        return total_distance

    def plot(self, projection='2D', output_file=None):
        "" "
        Visualize the LAB distribution using a 2D or 3D scatter plot.

        Parameters:
            projection (str): '2D' or '3D' for the type of plot. Default is '2D'.
            output_file (str, optional): File path to save the plot. If None, displays the plot interactively.
                                         Supports '.png' and '.tiff' formats.

        Raises:
            ValueError: If the model is unfitted or the projection type is unsupported.
        "" "
        if self.model is None:
            raise ValueError("The model must be fitted before plotting.")

        # Generate samples for visualization
        samples = self.sample(1000)

        # Create figure and axis based on the projection type
        if projection == '2D':
            fig, ax = plt.subplots(figsize=(8, 8))
            sc = ax.scatter(
                samples[:, 1],  # A-axis
                samples[:, 2],  # B-axis
                c=samples[:, 0],  # L channel as color
                cmap='viridis',
                s=10,
                alpha=0.6
            )
            plt.colorbar(sc, ax=ax, label="L (Lightness)")
            ax.set_xlabel('A (Red-Green)')
            ax.set_ylabel('B (Blue-Yellow)')
            ax.set_title('2D Visualization of LAB Distribution')

        elif projection == '3D':
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(
                samples[:, 0],  # L-axis
                samples[:, 1],  # A-axis
                samples[:, 2],  # B-axis
                c=samples[:, 0],  # L channel as color
                cmap='viridis',
                s=10,
                alpha=0.6
            )
            fig.colorbar(sc, ax=ax, label="L (Lightness)")
            ax.set_xlabel('L (Lightness)')
            ax.set_ylabel('A (Red-Green)')
            ax.set_zlabel('B (Blue-Yellow)')
            ax.set_title('3D Visualization of LAB Distribution')

        else:
            raise ValueError(f"Unsupported projection type: {projection}. Use '2D' or '3D'.")

        # Save or display the plot
        if output_file:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                plt.savefig(tmp_file.name, format="png")
                plt.close()

                temp_image = Image.open(tmp_file.name)
                if output_file.lower().endswith(".tiff"):
                    # Save as TIFF with sRGB profile
                    temp_image = ImageCms.profileToProfile(temp_image, self.plotting_utils.srgb_profile, self.plotting_utils.srgb_profile, outputMode="RGB")
                temp_image.save(output_file)
        else:
            plt.show()

    @property
    def mean(self):
        "" "
        Get the mean of the LAB distribution.

        Returns:
            numpy.ndarray: Mean of the distribution. Shape depends on the model:
                           - For GMM: (n_components, 3), where each row is the mean of a Gaussian component.
                           - For KDE: Single mean vector (3,).
            
        Raises:
            ValueError: If the model is not fitted before accessing this property.
        "" "
        if self.mean_ is None:
            raise ValueError("The model is not fitted. Mean is not available.")
        return self.mean_

    @property
    def covariance(self):
        "" "
        Get the covariance matrix of the LAB distribution.

        Returns:
            numpy.ndarray: Covariance matrix of the distribution. Shape depends on the model:
                           - For GMM: (n_components, 3, 3), where each matrix is the covariance of a Gaussian component.
                           - For KDE: Single covariance matrix (3, 3).
            
        Raises:
            ValueError: If the model is not fitted before accessing this property.
        "" "
        if self.covariance_ is None:
            raise ValueError("The model is not fitted. Covariance is not available.")
        return self.covariance_

    def save(self, file_path):
        "" "
        Save the distribution to a file.

        Parameters:
            file_path (str): Path to save the distribution object.
        
        Notes:
            The distribution is saved as a serialized pickle file.
        "" "
        if self.model is None:
            raise ValueError("The model is not fitted. Cannot save an unfitted distribution.")
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        "" "
        Load a distribution from a file.

        Parameters:
            file_path (str): Path to load the distribution object from.

        Returns:
            MultivariateLABDistribution: Loaded distribution object.
        
        Notes:
            The distribution is loaded from a serialized pickle file.
        "" "
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __str__(self):
        "" "
        String representation of the MultivariateLABDistribution object.

        Returns:
            str: Neatly formatted string showing the state of the distribution, 
                 including the model type, number of components, and key parameters.
        "" "
        if self.model is None:
            return "MultivariateLABDistribution (Unfitted)"
        else:
            summary = [
                "MultivariateLABDistribution (Fitted)",
                f"Method: GMM",
                f"Components: {len(self.weights_)}",
                f"Means: {np.array_str(self.mean_, precision=2, suppress_small=True)}",
                f"Weights: {np.array_str(self.weights_, precision=2, suppress_small=True)}",
                f"Covariance Shapes: {[cov.shape for cov in self.covariance_]}",
            ]
            return "\n".join(summary)
"""
