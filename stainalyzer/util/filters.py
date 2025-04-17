
"""
filters


"""

############################################################################################################
### Import
############################################################################################################

import io
import os
import cv2
import numpy as np
from sympy import Interval
from PIL import Image, ImageCms

from stainalyzer.util.utils import PlottingUtils, TripleInterval

############################################################################################################
### Classes
############################################################################################################

class DABFilters:
    """
    A class for applying various filters and transformations to an image, specifically for analyzing DAB-stained images.

    Parameters
    ----------
    image : np.ndarray
        The input image, typically read using OpenCV.
    mode : str, optional
        The color mode of the input image. Default is 'HSV'. Accepted values: 'HSV', 'BGR', 'RGB'.

    Attributes
    ----------
    image : np.ndarray
        The input image.
    mode : str
        The color mode of the input image.
    hsv_image : np.ndarray
        The image converted to HSV mode (if applicable).
    """

    def __init__(self):
        self.plotting_utils = PlottingUtils()

    def gamma_histogram_analysis(self, image: np.ndarray, color_space: str = "HSV") -> float:
        """
        Analyzes image brightness histogram and determines an adaptive gamma correction value.

        Parameters
        ----------
        image : numpy.ndarray
            Input image in HSV or LAB color space.
        color_space : str, optional
            The color space used ("HSV" or "LAB"). Default is "HSV".
        
        Returns
        -------
        float
            Recommended gamma value.
        """
        if color_space.upper() == "HSV":
            brightness_channel = image[:, :, 2]
        elif color_space.upper() == "LAB":
            brightness_channel = image[:, :, 0]
        else:
            raise ValueError("Invalid color_space. Choose 'HSV' or 'LAB'.")
        
        hist = cv2.calcHist([brightness_channel], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        mean_brightness = np.sum(hist * np.arange(256))
        
        gamma = -0.7 + (3 / (1 + ((mean_brightness/160)**1.0)))

        return gamma

    def gamma_correction(self, image: np.ndarray, gamma: float = 1.2, color_space: str = "HSV", output_file_name: str = None) -> np.ndarray:
        """
        Applies gamma correction to an image and plots histograms before and after correction.

        Parameters
        ----------
        image : numpy.ndarray
            Input image in BGR format.
        gamma : float, optional
            Gamma value for correction. Default is 1.2.
        color_space : str, optional
            The color space to use ("HSV" or "LAB"). Default is "HSV".
        output_file_name : str, optional
            Path to save the histogram plot. Default is None (no plot saved).

        Returns
        -------
        numpy.ndarray
            Gamma-corrected image in original input format.
        """
        if gamma <= 0:
            raise ValueError("Gamma must be > 0.")

        # Creates a copy of the original image
        converted_img = image.copy()  # Do not modify the original image

        # Gets the lightness channel for HSV and LAB
        if color_space.upper() == "HSV":
            brightness_channel = image[:, :, 2]
        elif color_space.upper() == "LAB":
            brightness_channel = image[:, :, 0]
        else:
            raise ValueError("Invalid color_space. Choose 'HSV' or 'LAB'.")

        # Creates histogram - before applying the filter
        hist_before, _ = np.histogram(brightness_channel.flatten(), bins=256, range=(0, 255))

        # Performs the correction based on the invert gamma distribution
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")

        # Use LUT algorithm
        corrected_brightness = cv2.LUT(brightness_channel, table)

        # Creates histogram - after applying the filter
        hist_after, _ = np.histogram(corrected_brightness.flatten(), bins=256, range=(0, 255))

        # Modify placeholder image to return the corrected version
        corrected_img = converted_img.copy()
        corrected_img[:, :, 0 if color_space.upper() == "LAB" else 2] = corrected_brightness

        # Prints the histograms of before and after
        if isinstance(output_file_name, str):
            self.plotting_utils.histogram([hist_before, hist_after], ["Original", "Corrected"], output_file_name)

        return corrected_img

    def alpha_histogram_analysis(self, image: np.ndarray, color_space: str = "HSV") -> float:
        """
        Analyzes the contrast of an image and determines an adaptive alpha correction value.

        Parameters
        ----------
        image : numpy.ndarray
            Input image in HSV or LAB color space.
        color_space : str, optional
            The color space used ("HSV" or "LAB"). Default is "HSV".

        Returns
        -------
        float
            Recommended alpha value for contrast enhancement.
        """
        # Ensure the provided color space is valid
        if color_space.upper() not in ["HSV", "LAB"]:
            raise ValueError("Invalid color_space. Choose 'HSV' or 'LAB'.")

        # Extract the appropriate brightness/intensity channel
        if color_space.upper() == "HSV":
            brightness_channel = image[:, :, 2]  # V channel in HSV
        else:
            brightness_channel = image[:, :, 0]  # L channel in LAB

        # Compute the standard deviation (contrast metric) of the brightness channel
        contrast = np.std(brightness_channel) / (np.mean(brightness_channel) + 1e-6)  # Avoid division by zero

        # Continuous mapping function for alpha correction based on contrast
        alpha = 1.0 + 0.5 * np.exp(-4 * contrast)  # Adaptive alpha mapping

        return alpha

    def alpha_correction(self, image: np.ndarray, alpha: float = 1.2, color_space: str = "HSV", output_file_name: str = None) -> np.ndarray:
        """
        Applies alpha correction to an image and plots histograms before and after correction.

        Parameters
        ----------
        image : numpy.ndarray
            Input image in BGR format.
        alpha : float, optional
            Alpha value for contrast correction. Default is 1.2.
        color_space : str, optional
            The color space to use ("HSV" or "LAB"). Default is "HSV".
        output_file_name : str, optional
            Path to save the histogram plot. Default is None (no plot saved).

        Returns
        -------
        numpy.ndarray
            Alpha-corrected image in its original input format.
        """
        # Ensure valid alpha
        if alpha <= 0:
            raise ValueError("Alpha must be > 0.")

        # Ensure the provided color space is valid
        if color_space.upper() not in ["HSV", "LAB"]:
            raise ValueError("Invalid color_space. Choose 'HSV' or 'LAB'.")

        # Creates a copy of the original image
        converted_img = image.copy()  # Do not modify the original image

        # Convert image to the selected color space
        if color_space.upper() == "HSV":
            brightness_channel = converted_img[:, :, 2]  # V channel in HSV
        else:
            brightness_channel = converted_img[:, :, 0]  # L channel in LAB

        # Compute histogram before correction
        hist_before, _ = np.histogram(brightness_channel.flatten(), bins=256, range=(0, 255))

        # Apply alpha correction using a linear scaling
        corrected_brightness = np.clip(alpha * brightness_channel, 0, 255)

        # Compute histogram after correction
        hist_after, _ = np.histogram(corrected_brightness.flatten(), bins=256, range=(0, 255))

        # Convert back to BGR color space
        corrected_img = converted_img.copy()
        corrected_img[:, :, 2 if color_space.upper() == "HSV" else 0] = corrected_brightness

        # Save the histogram plot if an output file is specified
        if isinstance(output_file_name, str):
            self.plotting_utils.histogram([hist_before, hist_after], ["Original", "Corrected"], output_file_name)

        return corrected_img

    def compress_sv(self, image, mode="HSV", alpha: float = 0.5, beta: float = 0.5) -> np.ndarray:
        """
        Compresses the S and V channels of the HSV image.

        Parameters
        ----------
        alpha : float, optional
            Compression factor for the V channel. Default is 0.5.
        beta : float, optional
            Compression factor for the S channel. Default is 0.5.

        Returns
        -------
        np.ndarray
            The resulting image in BGR format after compression.

        Raises
        ------
        ValueError
            If the image mode is not HSV.
        """
        if image is None:
            raise ValueError("Image is not in HSV mode. Cannot compress S and V channels.")

        H, S, V = cv2.split(image)
        S_mean = np.mean(S)
        V_mean = np.mean(V)

        S_compressed = S_mean + beta * (S - S_mean)
        V_compressed = V_mean + alpha * (V - V_mean)

        S_compressed = np.clip(S_compressed, 0, 255)
        V_compressed = np.clip(V_compressed, 0, 255)

        compressed_hsv = cv2.merge([H.astype(np.uint8), S_compressed.astype(np.uint8), V_compressed.astype(np.uint8)])
        return cv2.cvtColor(compressed_hsv, cv2.COLOR_HSV2BGR)

    def blur_sv(self, image, mode="HSV", kernel_size: tuple = (5, 5), sigmaX: float = 0, sigmaY: float = 0) -> np.ndarray:
        """
        Blurs the S and V channels of the HSV image using Gaussian Blur.

        Parameters
        ----------
        kernel_size : tuple, optional
            Kernel size for Gaussian blur. Default is (5, 5).
        sigmaX : float, optional
            Standard deviation in the X direction for Gaussian blur. Default is 0.
        sigmaY : float, optional
            Standard deviation in the Y direction for Gaussian blur. Default is 0.

        Returns
        -------
        np.ndarray
            The resulting image in BGR format after blurring.

        Raises
        ------
        ValueError
            If the image mode is not HSV.
        """
        if image is None:
            raise ValueError("Image is not in HSV mode. Cannot blur S and V channels.")

        # Remove H, S, V channels
        H, S, V = cv2.split(image)

        # Convert S and V channels to float32 for better precision during blurring
        S_blurred = cv2.GaussianBlur(S.astype(np.float32), kernel_size, sigmaX).astype(np.uint8)
        V_blurred = cv2.GaussianBlur(V.astype(np.float32), kernel_size, sigmaY).astype(np.uint8)

        # Merge channels back and convert to BGR
        blurred_hsv = cv2.merge([H, S_blurred, V_blurred])
        return cv2.cvtColor(blurred_hsv, cv2.COLOR_HSV2BGR)

    def nld_sv(self, image, mode="HSV", nonlinear_function: callable = None) -> np.ndarray:
        """
        Applies a nonlinear transformation to the S and V channels of the HSV image.

        Parameters
        ----------
        nonlinear_function : callable, optional
            A function to apply nonlinearity. Default is a sigmoid function.

        Returns
        -------
        np.ndarray
            The resulting image in BGR format after applying the transformation.

        Raises
        ------
        ValueError
            If the image mode is not HSV.
        """
        if image is None:
            raise ValueError("Image is not in HSV mode. Cannot apply nonlinear transformations to S and V channels.")

        if nonlinear_function is None:
            nonlinear_function = lambda x: 1 / (1 + np.exp(-0.1 * (x - 0.5)))  # Default sigmoid

        H, S, V = cv2.split(image)

        # Apply the nonlinear function to the S and V channels
        S_transformed = nonlinear_function(S.astype(np.float32) / 255) * 255  # Normalize to [0, 1] and rescale
        V_transformed = nonlinear_function(V.astype(np.float32) / 255) * 255

        # Clip and cast back to uint8
        S_transformed = np.clip(S_transformed, 0, 255).astype(np.uint8)
        V_transformed = np.clip(V_transformed, 0, 255).astype(np.uint8)

        # Merge channels and convert back to BGR
        transformed_hsv = cv2.merge([H, S_transformed, V_transformed])
        return cv2.cvtColor(transformed_hsv, cv2.COLOR_HSV2BGR)

    def __str__(self):
        return f"DABFilters()"

    def __repr__(self):
        return f"DABFilters()"

"""
    def apply_to_all_channels(self, image, mode="HSV", filter_function: callable, channel: int) -> np.ndarray:
        
        Applies a filter function to a specific channel (H/S/V or R/G/B).

        Parameters
        ----------
        filter_function : callable
            The function to apply to the selected channel.
        channel : int
            The channel to apply the filter to. For HSV: 0 = H, 1 = S, 2 = V.

        Returns
        -------
        np.ndarray
            The resulting image after applying the filter.

        Raises
        ------
        ValueError
            If the channel is invalid or the image mode does not match the filter.
       

        if image is None:
            raise ValueError("Image is not in HSV mode. Cannot apply filters to channels.")

        channels = cv2.split(image)
        if channel < 0 or channel >= len(channels):
            raise ValueError("Invalid channel index.")

        channels[channel] = filter_function(channels[channel])

        modified_hsv = cv2.merge(channels)
        return cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)
 """

class Mask:
    """
    Class to manage image masking operations, including weighted scoring,
    masked image generation, and applying masks to other images.
    
    Attributes
    ----------
    name : str
        Name of the mask.
    weight_tuple : tuple of float
        A tuple of weights that sum to 1.0, indicating the contribution of masked pixels.
    mask : TripleInterval
        An interval defining the masked-in pixels.
    image : numpy.ndarray
        The input image in OpenCV format.
    background_pixels : tuple of int, optional
        The color representing background pixels, defaults to HSV white (0, 0, 255).
    """

    def __init__(self, name, weight_tuple, mask, image, background_pixels=(0, 0, 255)):
        """
        Initializes the Mask object.

        Parameters
        ----------
        name : str
            Name of the mask.
        weight_tuple : tuple of float
            A tuple of weights that sum to 1.0, indicating the contribution of masked pixels.
        mask : TripleInterval
            An interval defining the masked-in pixels.
        image : numpy.ndarray
            The input image in OpenCV format.
        background_pixels : tuple of int, optional
            The color representing background pixels, defaults to HSV white (0, 0, 255).
        """

        if len(weight_tuple) != 3 or not np.isclose(sum(weight_tuple), 1.0):
            raise ValueError("weight_tuple must contain exactly three values that sum to 1.0")

        self.name = name
        self.weight_tuple = weight_tuple
        self.mask = mask
        self.image = image
        self.background_pixels = background_pixels

    def get_name(self):
        """
        Returns the current name.

        Returns
        -------
        str
            The stored name.
        """
        return self.name

    def set_mask(self, name):
        """
        Sets a new name.

        Parameters
        ----------
        name : str
            The new name.
        """
        self.name = name

    def get_weight_tuple(self):
        """
        Returns the weight tuple.

        Returns
        -------
        tuple of float
            The weight tuple used for scoring.
        """
        return self.weight_tuple

    def set_weight_tuple(self, weight_tuple):
        """
        Sets a new weight tuple.

        Parameters
        ----------
        weight_tuple : tuple of float
            A tuple of three values that sum to 1.0.

        Raises
        ------
        ValueError
            If the tuple does not contain exactly three values that sum to 1.0.
        """
        if len(weight_tuple) != 3 or not np.isclose(sum(weight_tuple), 1.0):
            raise ValueError("weight_tuple must contain exactly three values that sum to 1.0")
        self.weight_tuple = weight_tuple

    def get_mask(self):
        """
        Returns the current TripleInterval mask.

        Returns
        -------
        TripleInterval
            The stored mask.
        """
        return self.mask

    def set_mask(self, mask):
        """
        Sets a new TripleInterval mask and updates the image mask if an image exists.

        Parameters
        ----------
        mask : TripleInterval
            The new mask to be applied.
        """
        self.mask = mask

    def get_image(self):
        """
        Returns the stored image.

        Returns
        -------
        numpy.ndarray
            The stored OpenCV image.
        """
        return self.image

    def set_image(self, image):
        """
        Sets a new image and updates the mask.

        Parameters
        ----------
        image : numpy.ndarray
            The new OpenCV image to be used.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy ndarray")
        
        self.image = image

    def get_background_pixels(self):
        """
        Returns the background pixel.

        Returns
        -------
        tuple
            The stored background pixel.
        """
        return self.background_pixels

    def set_background_pixels(self, background_pixels):
        """
        Sets a new background pixel.

        Parameters
        ----------
        background_pixels : tuple
            The new background pixel.
        """
        self.background_pixels = background_pixels

    def weighted_score(self):
        """
        Computes a weighted score based on the number of non-background, masked-in pixels.

        Returns
        -------
        tuple of float
            Weighted pixel count based on `self.weight_tuple`.
        """
        non_bg_mask = np.any(self.image != self.background_pixels, axis=-1)
        mask_applied = np.zeros_like(non_bg_mask, dtype=np.uint8)

        for start, end in self.mask.intervals:
            mask_segment = cv2.inRange(self.image, np.array(start, dtype=np.uint8), np.array(end, dtype=np.uint8))
            mask_applied = cv2.bitwise_or(mask_applied, mask_segment)

        masked_pixels = np.count_nonzero(mask_applied & non_bg_mask)

        return tuple(w * masked_pixels for w in self.weight_tuple)

    def masked_image(self):
        """
        Returns an image with masked-in pixels preserved and background pixels filled.

        Returns
        -------
        numpy.ndarray
            The image with masked-in pixels and background filled.
        """
        non_bg_mask = np.any(self.image != self.background_pixels, axis=-1)
        mask_applied = np.zeros_like(non_bg_mask, dtype=np.uint8)

        for start, end in self.mask.intervals:
            mask_segment = cv2.inRange(self.image, np.array(start, dtype=np.uint8), np.array(end, dtype=np.uint8))
            mask_applied = cv2.bitwise_or(mask_applied, mask_segment)

        final_mask = mask_applied & non_bg_mask
        result_image = np.full_like(self.image, self.background_pixels, dtype=self.image.dtype)
        result_image[final_mask > 0] = self.image[final_mask > 0]

        return result_image

    def put(self, another_image, proportion=1.0):
        """
        Places a proportion of masked-in pixels from this mask into another image.

        Parameters
        ----------
        another_image : numpy.ndarray
            The target image where the masked pixels will be placed.
        proportion : float, optional
            The proportion of masked pixels to put into another_image (0.0 to 1.0).
        """
        if proportion == 0.0:
            return

        non_bg_mask = np.any(self.image != self.background_pixels, axis=-1)
        mask_applied = np.zeros_like(non_bg_mask, dtype=np.uint8)

        for start, end in self.mask.intervals:
            mask_segment = cv2.inRange(self.image, np.array(start, dtype=np.uint8), np.array(end, dtype=np.uint8))
            mask_applied = cv2.bitwise_or(mask_applied, mask_segment)

        final_mask = mask_applied & non_bg_mask
        masked_indices = np.argwhere(final_mask)

        num_pixels_to_replace = int(proportion * len(masked_indices))

        if num_pixels_to_replace > 0:
            selected_indices = masked_indices[np.random.choice(len(masked_indices), num_pixels_to_replace, replace=False)]
            for y, x in selected_indices:
                another_image[y, x] = self.image[y, x]

        return

    @staticmethod
    def compute_mean(image):
        """
        Computes the mean color of an image, excluding pixels with 0 or 255 in any channel.

        Parameters
        ----------
        image : numpy.ndarray
            The input image in OpenCV format.

        Returns
        -------
        numpy.ndarray
            The mean color values for each channel.
        """

        # Create a mask to exclude pixels with any 0 or 255 in any channel
        valid_pixels_mask = ~np.any((image == 0) | (image == 255), axis=-1)
        valid_pixels = image[valid_pixels_mask]

        if valid_pixels.size > 0:
            mean_color = np.mean(valid_pixels, axis=0)
        else:
            mean_color = np.array([0, 0, 0])  # Default if no valid pixels are found

        return mean_color

    @staticmethod
    def compute_covariance(image):
        """
        Computes the covariance matrix of an image, excluding pixels with 0 or 255 in any channel.

        Parameters
        ----------
        image : numpy.ndarray
            The input image in OpenCV format.

        Returns
        -------
        numpy.ndarray
            The covariance matrix of the color channels.
        """
        # Create a mask to exclude pixels with any 0 or 255 in any channel
        valid_pixels_mask = ~np.any((image == 0) | (image == 255), axis=-1)
        valid_pixels = image[valid_pixels_mask]

        if valid_pixels.size > 0:
            covariance_matrix = np.cov(valid_pixels, rowvar=False)
        else:
            covariance_matrix = np.zeros((3, 3))  # Default if no valid pixels are found

        return covariance_matrix
