
"""
Pre-processing

"""

############################################################################################################
### Import
############################################################################################################

import io
import os
import cv2
import numpy as np
from math import ceil
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from skimage.segmentation import slic

############################################################################################################
### Constants
############################################################################################################

# Constants
SEED = 1987
np.random.seed(SEED)

############################################################################################################
### Multivariate Gaussian Class
############################################################################################################

class EnhancementPreprocessor:
    """
    A class for preprocessing images, including CLAHE normalization, dynamic thresholding,
    SLIC segmentation, and K-means clustering.

    Attributes:
        input_image (str): Path to the input image or a "fantasy path" for preloaded images.
        original_image (numpy.ndarray): The original image (either loaded from a file or provided directly).
        lab_image (numpy.ndarray): CLAHE-normalized LAB image.
        dynamic_threshold (float): Calculated dynamic threshold.
        superpixel_segments (numpy.ndarray): Output of SLIC segmentation.
        quantized_image (numpy.ndarray): Output of K-means clustering.
        centroids (numpy.ndarray): Cluster centroids.
        pixel_counts (numpy.ndarray): Pixel counts per cluster.
        cluster_labels (numpy.ndarray): Cluster labels from K-means.
    """

    def __init__(self, input_image, replace_black_param=None, clahe_params=None, slic_params=None, kmeans_params=None):
        """
        Initialize the ImagePreprocessor class.

        Parameters:
            input_image (str or numpy.ndarray): Path to the input image or a preloaded OpenCV image.
            replace_black_param (dict, optional): Parameters for replacing black areas in the image.
            clahe_params (dict, optional): Parameters for CLAHE normalization.
            slic_params (dict, optional): Parameters for SLIC segmentation.
            kmeans_params (dict, optional): Parameters for K-means clustering.

        Raises:
            ValueError: If `input_image` is not a string or a numpy.ndarray.
        """

        # Initialize original image
        if isinstance(input_image, str):
            # Load the image from the given path
            self.input_image = input_image
            self.original_image = cv2.imread(input_image)
            if self.original_image is None:
                raise ValueError(f"Unable to load image from the path: {input_image}")
        elif isinstance(input_image, np.ndarray):
            # Use the provided image
            self.input_image = "<preloaded_image>"
            self.original_image = input_image
        else:
            raise ValueError("'input_image' must be a string (file path) or a numpy.ndarray (preloaded image).")

        # Initialize attributes for processing steps
        self.lab_image = None
        self.dynamic_threshold = None
        self.superpixel_segments = None
        self.quantized_image = None
        self.centroids = None
        self.pixel_counts = None
        self.cluster_labels = None
        self.processed_image = None
        self.dynamic_threshold = None
        self.superpixel_segments = None
        self.quantized_image = None
        self.centroids = None
        self.pixel_counts = None
        self.cluster_labels = None
        self._replace_black_param = replace_black_param or {'black_threshold': 3, 'color_space': "BGR"}
        self._clahe_params = clahe_params or {'color_scheme': "LAB", 'clipLimit': 2.0, 'tileGridSize': (8, 8), 'return_threshold':False}
        self._slic_params = slic_params or {'n_segments': 250, 'compactness': 10, 'start_label': 0}
        self._kmeans_params = kmeans_params or {'n_clusters': 10, 'random_state': 42}

    # Full preprocessingpipeline
    def preprocess(self):
        """Run the full preprocessing pipeline."""
        self.replace_black_pixels(black_threshold=self._replace_black_param["black_threshold"])
        self.apply_clahe(return_threshold=True)
        self.apply_slic(
                        n_segments = self.dynamic_threshold * self._slic_params["n_segments"],
                        compactness = self.dynamic_threshold,
                        )
        self.apply_kmeans(
                          n_clusters = np.uint8(ceil(self.dynamic_threshold/2)),
                          use_superpixels=True
                          )

    # Replace Black Pixels By Closest Non-black Neighbor
    def replace_black_pixels(self, black_threshold=None, color_space=None):
        """
        Replace black pixels in self.original_image with the nearest non-black pixel in the specified color space.

        Raises:
            ValueError: If the self.original_image contains only black pixels.
        """

        # Parameter processing
        if color_space is None:
            black_threshold = self._replace_black_param["black_threshold"]
            color_space = self._replace_black_param["color_space"]

        # Convert self.original_image to the specified color space
        if color_space.upper() == "BGR":
            converted_image = self.original_image
        elif color_space.upper() == "RGB":
            converted_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        elif color_space.upper() == "HSV":
            converted_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        elif color_space.upper() == "LAB":
            converted_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)
        else:
            raise ValueError(
                            (f"Unsupported color space: {color_space}."),
                            (f"Use 'BGR', 'RGB', 'HSV', or 'LAB'.")
                            )

        # Create a mask where pixels are considered black if all BGR values are <= black_threshold
        black_mask = np.all(converted_image <= black_threshold, axis=-1)

        # Get black and non-black coordinates
        black_coords = np.argwhere(black_mask)
        non_black_coords = np.argwhere(~black_mask)

        # Extract the non-black pixel values
        non_black_pixels = converted_image[~black_mask]

        # Handle case where the image contains only black pixels
        if len(non_black_coords) == 0:
            raise ValueError("Image contains only black pixels.")

        # Create KDTree for fast nearest-neighbor search
        tree = KDTree(non_black_coords)

        # Replace black pixels with the nearest non-black pixel
        for x, y in black_coords:
            nearest_idx = tree.query([x, y])[1]
            converted_image[x, y] = non_black_pixels[nearest_idx]

        # Convert back to BGR if necessary
        if color_space.upper() == "BGR":
            self.original_image = converted_image
        elif color_space.upper() == "HSV":
            self.original_image = cv2.cvtColor(converted_image, cv2.COLOR_HSV2BGR)
        elif color_space.upper() == "LAB":
            self.original_image = cv2.cvtColor(converted_image, cv2.COLOR_LAB2BGR)
        elif color_space.upper() == "RGB":
            self.original_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2BGR)

    # CLAHE Normalization & Dynamic Threshold
    def apply_clahe(self, color_scheme=None, clipLimit=None, tileGridSize=None, return_threshold=True):
        """
        Perform CLAHE normalization on self.original_image in the specified color scheme.
        (optionally) Calculates a dynamic threshold.

        Parameters:
            return_threshold (bool): Whether to calculate and return the dynamic threshold.
        """

        # Parameter processing
        if color_scheme is None:
            color_scheme = self._clahe_params["color_scheme"]
        if clipLimit is None:
            clipLimit = self._clahe_params["clipLimit"]
        if tileGridSize is None:
            tileGridSize = self._clahe_params["tileGridSize"]
        if return_threshold is None:
            return_threshold = self._clahe_params["return_threshold"]

        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

        if color_scheme.upper() == "LAB":

            # Convert BGR to LAB
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)

            # Apply CLAHE to the Lightness channel
            self.processed_image[..., 0] = clahe.apply(self.processed_image[..., 0])

            if return_threshold:
                # Calculate threshold using the L channel
                self.dynamic_threshold = int(np.std(self.processed_image[..., 0]))

        elif color_scheme.upper() == "HSV":

            # Convert BGR to HSV
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)

            # Apply CLAHE to the Value channel
            self.processed_image[..., 2] = clahe.apply(self.processed_image[..., 2])

            if return_threshold:

                # Calculate threshold using the V channel
                self.dynamic_threshold = int(np.std(self.processed_image[..., 2]))

        elif color_scheme.upper() == "BGR":

            # Split channels
            b, g, r = cv2.split(self.original_image)

            # Apply CLAHE to each channel
            b = clahe.apply(b)
            g = clahe.apply(g)
            r = clahe.apply(r)

            self.processed_image = cv2.merge((b, g, r))

            if return_threshold:

                # Calculate threshold using the average of all channels
                self.dynamic_threshold = int(np.std(merged_image))


        else:
            raise ValueError(f"Unsupported color scheme: {color_scheme}. Use 'BGR', 'LAB', or 'HSV'.")

    # Perform Superpixel Segmentation using SLIC
    def apply_slic(self, n_segments=None, compactness=None, start_label=None):
        """
        Perform superpixel segmentation on self.processed_image using the SLIC algorithm.

        Raises:
            ValueError: If 'self.processed_image' does not have 3 channels.
        """

        # Parameter processing
        if n_segments is None:
            n_segments = self._slic_params["n_segments"]
        if compactness is None:
            compactness = self._slic_params["compactness"]
        if start_label is None:
            start_label = self._slic_params["start_label"]

        # Ensure the image is in LAB color space for better segmentation
        if self.processed_image.shape[-1] != 3:
            raise ValueError("Input image must have three color channels (e.g., LAB format).")

        # Perform SLIC segmentation
        self.superpixel_segments = slic(
                                        self.processed_image,
                                        n_segments=n_segments,
                                        compactness=compactness,
                                        start_label=start_label
                                        )

    # Perform K-Means Clustering
    def apply_kmeans(self, n_clusters=None, random_state=None, use_superpixels=False):
        """
        Perform K-Means clustering on a CLAHE-normalized self.processed_image, with optional superpixel-based initialization.

        Parameters:
            use_superpixels (bool): If True, initialize cluster centroids using superpixel segment means. Default is False.

        Raises:
            ValueError: If `use_superpixels=True` and `segments` is not provided.
        """

        # Parameter processing
        if n_clusters is None:
            n_clusters = self._kmeans_params["n_clusters"]
        if random_state is None:
            random_state = self._kmeans_params["random_state"]

        # Flatten LAB image into a 2D array of pixels
        pixels = self.processed_image.reshape(-1, 3)

        # Initialize centroids using superpixels if specified
        if use_superpixels:
            if self.superpixel_segments is None:
                raise ValueError("Superpixel segments must be provided when use_superpixels=True.")
            
            # Compute mean LAB color for each superpixel
            unique_segments = np.unique(self.superpixel_segments)
            initial_centroids = []
            for seg_val in unique_segments:
                mask = self.superpixel_segments == seg_val
                mean_color = self.processed_image[mask].mean(axis=0)  # Calculate mean LAB for the self.superpixel_segments
                initial_centroids.append(mean_color)

            initial_centroids = np.array(initial_centroids, dtype=np.float32)  # Convert to NumPy array
            
            # Limit initial centroids to `num_clusters`
            if len(initial_centroids) > n_clusters:
                initial_centroids = initial_centroids[:n_clusters]

            # Perform K-means clustering with pre-defined centroids
            kmeans = KMeans(
                n_clusters=len(initial_centroids),
                init=initial_centroids,
                n_init=1,
                random_state=random_state
            )
        else:
            # Global K-means without superpixel initialization
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

        # Fit the model and predict cluster labels
        labels = kmeans.fit_predict(pixels)

        # Reshaping labels
        height, width, _ = self.processed_image.shape
        self.cluster_labels = labels.reshape((height, width))

        # Generate quantized self.processed_image from cluster labels
        quantized_pixels = np.round(kmeans.cluster_centers_).astype("uint8")[labels]
        self.quantized_image = quantized_pixels.reshape(self.processed_image.shape)

        # Extract centroids and pixel counts
        self.centroids = np.round(kmeans.cluster_centers_).astype("uint8")
        self.pixel_counts = np.bincount(labels, minlength=n_clusters)

    @property
    def replace_black_param(self):
        """
        Get the Replace black parameters dictionary.

        Returns:
            dict: Replace black parameters dictionary.
            
        Raises:
            ValueError: If Replace black parameters dictionary does not exist.
        """
        if self._replace_black_param is None:
            raise ValueError("Replace black parameters dictionary is not available.")
        return self._replace_black_param

    @replace_black_param.setter
    def replace_black_param(self, value):
        """
        Set the Replace black parameters dictionary.

        Parameters:
            value (dict): Dictionary containing Replace black parameters.
        
        Raises:
            TypeError: If the provided value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("Replace black parameters must be a dictionary.")
        self._replace_black_param = value

    @property
    def clahe_params(self):
        """
        Get the CLAHE parameters dictionary.

        Returns:
            dict: CLAHE parameters dictionary.
            
        Raises:
            ValueError: If CLAHE parameters dictionary does not exist.
        """
        if self._clahe_params is None:
            raise ValueError("CLAHE parameters dictionary is not available.")
        return self._clahe_params

    @clahe_params.setter
    def clahe_params(self, value):
        """
        Set the CLAHE parameters dictionary.

        Parameters:
            value (dict): Dictionary containing CLAHE parameters.
        
        Raises:
            TypeError: If the provided value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("CLAHE parameters must be a dictionary.")
        self._clahe_params = value

    @property
    def slic_params(self):
        """
        Get the SLIC parameters dictionary.

        Returns:
            dict: SLIC parameters dictionary.
            
        Raises:
            ValueError: If SLIC parameters dictionary does not exist.
        """
        if self._slic_params is None:
            raise ValueError("SLIC parameters dictionary is not available.")
        return self._slic_params

    @slic_params.setter
    def slic_params(self, value):
        """
        Set the SLIC parameters dictionary.

        Parameters:
            value (dict): Dictionary containing SLIC parameters.
        
        Raises:
            TypeError: If the provided value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("SLIC parameters must be a dictionary.")
        self._slic_params = value

    @property
    def kmeans_params(self):
        """
        Get the K-means parameters dictionary.

        Returns:
            dict: K-means parameters dictionary.
            
        Raises:
            ValueError: If K-means parameters dictionary does not exist.
        """
        if self._kmeans_params is None:
            raise ValueError("K-means parameters dictionary is not available.")
        return self._kmeans_params

    @kmeans_params.setter
    def kmeans_params(self, value):
        """
        Set the K-means parameters dictionary.

        Parameters:
            value (dict): Dictionary containing K-means parameters.
        
        Raises:
            TypeError: If the provided value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("K-means parameters must be a dictionary.")
        self._kmeans_params = value
