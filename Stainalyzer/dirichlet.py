
"""
dirichlet

Implements a Dirichlet Process Mixture Model for Image Quantization

1. Instantiate the Model:

dpmm = DirichletProcessMixtureModel(n_components=50, random_state=42)

2. Fit the Model:

dpmm.fit(image_data)

3. Quantize an Image:

quantized_image = dpmm.quantize_image(image, dimensions=[0, 1, 2])

4. Visualize Clusters:

dpmm.plot_clusters(image_data, dimensions=[0, 1])

----------------------

Constructor:

# Example Usage:
# Assuming `data` is a 2D array and `image` is a 3D image array:
# dpmm = DirichletProcessMixtureModel(n_components=50, covariance_type='diag', random_state=42)
# dpmm.fit(data)
# quantized_image = dpmm.quantize_image(image, [0, 1, 2])

Fit:

# Example Usage:
# Assuming `SomeDirichletProcessGMM` is a valid DPGMM implementation that
# provides fit, predict, means_, and covariances_ methods:
#
# from some_library import SomeDirichletProcessGMM
# dpgmm_model = SomeDirichletProcessGMM()
# dpmm = DirichletProcessMixtureModel(model=dpgmm_model)
# data = np.random.rand(100, 2)  # 100 samples, 2 features
# dpmm.fit(data)

Quantize Image:

# Example Usage:
# Assuming `SomeDirichletProcessGMM` is a valid DPGMM implementation that
# provides fit, predict, means_, and covariances_ methods:
#
# from some_library import SomeDirichletProcessGMM
# dpgmm_model = SomeDirichletProcessGMM()
# dpmm = DirichletProcessMixtureModel(model=dpgmm_model)
# data = np.random.rand(100, 2)  # 100 samples, 2 features
# dpmm.fit(data)
# image = np.random.rand(100, 100, 3)  # Example image
# quantized_image = dpmm.quantize_image(image, [0, 1, 2])

Save Model / Load Model:

# Example Usage:
# Assuming `data` is a 2D array and `image` is a 3D image array:
# dpmm = DirichletProcessMixtureModel(n_components=50, covariance_type='diag', random_state=42)
# dpmm.fit(data)
# dpmm.save_model("dpmm_model.pkl")
# loaded_dpmm = DirichletProcessMixtureModel.load_model("dpmm_model.pkl")
# quantized_image = loaded_dpmm.quantize_image(image, [0, 1, 2])

"""

############################################################################################################
### Import
############################################################################################################

import os
import cv2
import math
import pickle
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

############################################################################################################
### Classes
############################################################################################################

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import pickle

class DirichletProcessMixtureModel:
    """
    A class implementing the Dirichlet Process Mixture Model (DPMM) for image quantization.

    Parameters
    ----------
    n_components : int, optional
        Maximum number of clusters for the mixture model. Defaults to 100.
    covariance_type : str, optional
        Type of covariance matrices for the Gaussian components. Can be 'full', 'tied', 'diag', or 'spherical'.
        Defaults to 'full'.
    random_state : int, optional
        Random seed for reproducibility. Defaults to None.
    max_iter : int, optional
        Maximum number of iterations for the Expectation-Maximization (EM) algorithm. Defaults to 100.
    tol : float, optional
        Convergence threshold for the EM algorithm. Defaults to 1e-3.
    n_init : int, optional
        Number of initializations to perform. Defaults to 1.
    init_params : str, optional
        Method used to initialize the weights, means, and precisions. Can be 'kmeans' or 'random'. Defaults to 'kmeans'.
    weight_concentration_prior : float, optional
        The Dirichlet Process prior for cluster weights. Controls the expected number of clusters. Defaults to None.
    reg_covar : float, optional
        Non-negative regularization added to the diagonal of covariance matrices to ensure numerical stability. Defaults to 1e-6.
    verbose : int, optional
        Verbosity level. Higher values print more convergence details. Defaults to 0.

    Attributes
    ----------
    model : BayesianGaussianMixture
        The underlying DPMM model.
    labels_ : np.ndarray
        Cluster labels for each data point.
    means_ : np.ndarray
        Mean of each Gaussian component.
    covariances_ : np.ndarray
        Covariance matrices of each Gaussian component.
    """

    def __init__(self, n_components=100, covariance_type='full', random_state=None,
                 max_iter=100, tol=1e-3, n_init=1, init_params='kmeans',
                 weight_concentration_prior=None, reg_covar=1e-6, verbose=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init_params = init_params
        self.weight_concentration_prior = weight_concentration_prior
        self.reg_covar = reg_covar
        self.verbose = verbose

        self.model = BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=self.weight_concentration_prior,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            init_params=self.init_params,
            reg_covar=self.reg_covar,
            verbose=self.verbose,
            random_state=self.random_state
        )

        self.labels_ = None
        self.means_ = None
        self.covariances_ = None

    def fit(self, data):
        """
        Fits the Dirichlet Process Mixture Model to the data.

        This method trains the model on the provided dataset and computes cluster labels,
        cluster means, and covariances for the resulting mixture components.

        Parameters
        ----------
        data : np.ndarray
            The data to fit. Shape: (n_samples, n_features), where n_samples is the number of
            samples and n_features corresponds to the dimensionality of the data.

        Returns
        -------
        None

        Attributes Updated
        ------------------
        labels_ : np.ndarray
            Array of cluster labels assigned to each data point. Shape: (n_samples,).
        
        means_ : np.ndarray
            Means of the fitted Gaussian components. Shape: (n_components, n_features),
            where n_components is the number of components identified by the model.
        
        covariances_ : np.ndarray
            Covariance matrices of the fitted Gaussian components. Shape: (n_components, n_features, n_features).
        """
        # Fit the model to the data
        self.model.fit(data)

        # Predict cluster labels for each data point
        self.labels_ = self.model.predict(data)

        # Extract the means of the fitted components
        self.means_ = self.model.means_

        # Extract the covariance matrices of the fitted components
        self.covariances_ = self.model.covariances_

    def quantize_image(self, image, dimensions):
        """
        Quantizes an image using the fitted DPMM model.

        Parameters
        ----------
        image : np.ndarray
            The input image in BGR or RGB format. Shape: (height, width, channels).
        dimensions : list of int
            The dimensions of the color space to use (e.g., [0, 1, 2] for RGB).

        Returns
        -------
        quantized_image : np.ndarray
            The quantized image with the same shape as the input image.

        Raises
        ------
        ValueError
            If the model has not been fitted before calling this method.
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted before quantizing an image.")

        # Flatten the image into a 2D array: (n_pixels, n_features)
        pixels = image.reshape(-1, image.shape[2])[:, dimensions]

        # Predict labels for each pixel
        labels = self.model.predict(pixels)

        # Replace pixel values with the mean of their assigned cluster
        quantized_pixels = self.means_[labels]

        # Reshape back to the original image shape
        quantized_image = quantized_pixels.reshape(image.shape)

        # Ensure the output image is in uint8 format
        quantized_image = np.clip(quantized_image, 0, 255).astype(np.uint8)

        return quantized_image


    def save_model(self, filepath):
        """
        Saves the fitted DPMM model to a file.

        Parameters
        ----------
        filepath : str
            Path to the file where the model will be saved.

        Returns
        -------
        None
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath):
        """
        Loads a DPMM model from a file.

        Parameters
        ----------
        filepath : str
            Path to the file where the model is saved.

        Returns
        -------
        DirichletProcessMixtureModel
            The loaded DPMM model.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def score(self, data):
        """
        Computes the log-likelihood of the data under the fitted model.

        Parameters
        ----------
        data : np.ndarray
            The data to evaluate. Shape: (n_samples, n_features).

        Returns
        -------
        float
            The log-likelihood of the data under the model.
        """
        return self.model.score(data)

    def reset(self):
        """
        Resets the model, clearing all fitted attributes.

        Returns
        -------
        None
        """
        self.model = BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            weight_concentration_prior_type='dirichlet_process',
            random_state=self.random_state
        )
        self.labels_ = None
        self.means_ = None
        self.covariances_ = None

    def is_fitted(self):
        """
        Checks if the model has been fitted.
        
        Returns
        -------
        bool
            True if the model is fitted, False otherwise.
        """
        return self.labels_ is not None and self.means_ is not None and self.covariances_ is not None

    def transform(self, data):
        """
        Transforms the input data by replacing each point with its cluster mean.
        
        Parameters
        ----------
        data : np.ndarray
            The data to transform. Shape: (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Transformed data where each point is replaced with its cluster mean.
        """
        if not self.is_fitted():
            raise ValueError("Model must be fitted before transforming data.")
        
        labels = self.model.predict(data)
        transformed_data = self.means_[labels]
        return transformed_data

    def get_cluster_info(self):
        """
        Retrieves information about the clusters, including the number of clusters,
        their means, and covariance matrices.
        
        Returns
        -------
        dict
            A dictionary with cluster information:
            - 'n_clusters': Number of clusters.
            - 'means': Means of the Gaussian components.
            - 'covariances': Covariance matrices of the Gaussian components.
        """
        if not self.is_fitted():
            raise ValueError("Model must be fitted before retrieving cluster information.")
        
        return {
            'n_clusters': len(self.means_),
            'means': self.means_,
            'covariances': self.covariances_
        }

    def __str__(self):
        """
        Returns a string representation of the DirichletProcessMixtureModel instance.
        
        Returns
        -------
        str
            A user-friendly description of the model.
        """
        fitted_status = "Fitted" if self.is_fitted() else "Not Fitted"
        return (f"DirichletProcessMixtureModel(n_components={self.n_components}, "
                f"covariance_type='{self.covariance_type}', random_state={self.random_state}, "
                f"status={fitted_status})")

    def __repr__(self):
        """
        Returns a detailed string representation of the DirichletProcessMixtureModel instance.
        
        Returns
        -------
        str
            A detailed string representation of the model for debugging.
        """
        return (f"DirichletProcessMixtureModel(n_components={self.n_components}, "
                f"covariance_type='{self.covariance_type}', random_state={self.random_state})")



