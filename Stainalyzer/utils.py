
"""
Utilities
"""

############################################################################################################
### Import
############################################################################################################

import io
import os
import cv2
import math
import struct
import colour
import tempfile
import argparse
import numpy as np
import seaborn as sns
from math import ceil
from typing import List, Tuple
from PIL import Image, ImageCms
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter1d

############################################################################################################
### Classes
############################################################################################################

class ColorName:
    """
    A class to store and provide color names.

    TODO

    Attributes:
        TODO
    """

    def __init__(self, file_path=None):
        """
        Initialize the MultivariateLABDistribution class.

        Parameters:
            file_path (str): TODO

        Attributes Initialized:
            color_names (dict): A dictionary between hexadecimal colors and color names.

        Raises:
            ValueError: If the provided `method` is not supported.

        Example Usage:
            TODO
        """
        # Initialize attributes
        self.file_path = file_path
        self._color_names = None

        # Automatically load color names from file
        if file_path is not None:
            self.load_hexadecimal_to_name(file_path=self.file_path)

    # Load File with Hexadecimal Color Names from Community
    def load_hexadecimal_to_name(self, file_path):
        """
        Load the hexadecimal-to-name mapping from a community color-vote csv-table file.
        Args:
            file_path (str): Path to the color names file.
        Returns:
            dict: A dictionary mapping hexadecimal color codes to their names.
        """
        self._color_names = {}
        with open(file_path, 'r') as f:
            f.readline()
            for line in f:
                # Split line into columns
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                hex_color, name = parts[0], parts[1]
                self._color_names[hex_color] = name

    # Get closest color name
    def closest_color_name(self, rgb_color):
        """
        Find the closest named color to the given RGB color.
        Args:
            rgb_color (tuple): A tuple representing the RGB color (R, G, B).
        Returns:
            str: The name of the closest color.
        """

        # Convert RGB to lowercase hexadecimal
        hex_color = f"{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"
        
        # Exact match
        if hex_color in self._color_names:
            return self._color_names[hex_color]

        # Zigzag search for the closest color
        min_distance = float('inf')
        closest_name = None
        for hex_key, name in self._color_names.items():

            # Convert hex_key to RGB (float otherwise warning)
            r = float(int(hex_key[0:2], 16))
            g = float(int(hex_key[2:4], 16))
            b = float(int(hex_key[4:6], 16))

            # Calculate Manhattan distance
            distance = self.calculate_color_distance((r, g, b), rgb_color)
            if distance < min_distance:
                min_distance = distance
                closest_name = name

        return closest_name

    # Calculate color distance
    def calculate_color_distance(self, color1, color2, mode='manhattan'):
        """
        Calculate the manhattan or euclidean distance between two colors.

        Args:
            color1 (tuple): A tuple representing the RGB color (R, G, B).
            color2 (tuple): A tuple representing the RGB color (R, G, B).
            mode (str): The type of distance. Can be {'manhattan', 'euclidean'}

        Returns:
            str: The manhattan or euclidean distance between color1 and color2.
        """

        if mode == 'manhattan':
            return sum(abs(c1 - c2) for c1, c2 in zip(color1, color2))
        elif mode == 'euclidean':
            return sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)) ** 0.5
        else:
            raise ValueError("Unsupported mode. Use 'manhattan' or 'euclidean'.")

    @property
    def color_names(self):
        """
        Get the color_names dictionary.

        Returns:
            dict: color_names dictionary.
            
        Raises:
            ValueError: If color_names dictionary does not exist.
        """
        if self._color_names is None:
            raise ValueError("Color names dictionary is not available.")
        return self._color_names

    @color_names.setter
    def color_names(self, value):
        """
        Set the color_names dictionary.

        Parameters:
            value (dict): Dictionary containing hexadecimal to color names.
        
        Raises:
            TypeError: If the provided value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("color_names must be a dictionary.")
        self._color_names = value

class PlottingUtils:
    """
    A utility class for centralized management of plotting and saving images with standardized sRGB profiles.

    This class provides tools for working with images, including managing color profiles
    (e.g., sRGB) and standardizing plotting-related operations.
    This class supports saving images as TIFFs with proper ICC profiles and handling common visualization tasks.
    """

    def __init__(self, srgb_profile=None):
        """
        Initialize the PlottingUtils class.

        Parameters:
            srgb_profile (object, optional): A color profile for image processing. 
                                             Defaults to an sRGB profile if not provided.
        """
        if srgb_profile is None:
            self.srgb_profile = ImageCms.createProfile("sRGB")
        else:
            self.srgb_profile = srgb_profile

    @property
    def srgb_profile(self):
        """
        Get the sRGB profile.

        Returns:
            object: The currently set sRGB profile.
        """
        return self._srgb_profile

    @srgb_profile.setter
    def srgb_profile(self, profile):
        """
        Set the sRGB profile.

        Parameters:
            profile (object): The new sRGB profile to use.

        Raises:
            TypeError: If the provided profile is not a valid ICC profile object.
        """
        if not isinstance(profile, ImageCms.core.CmsProfile):
            raise TypeError("The srgb_profile must be a valid ICC profile object.")
        self._srgb_profile = profile

    def save_as_tiff(self, image, file_path):
        """
        Save an image as a TIFF with the sRGB profile.

        Parameters:
            image (numpy.ndarray or PIL.Image.Image): The image to save.
            file_path (str): Path to save the TIFF image.
        """
        # Convert OpenCV image to PIL if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Apply sRGB profile and save as TIFF
        image = ImageCms.profileToProfile(image, self.srgb_profile, self.srgb_profile, outputMode="RGB")
        image.save(file_path, format="TIFF")

    def save_picture(self, output_file_name, *images, n_rows=1, n_cols=1, color_space="BGR", 
                     output_format="TIFF", plot_grids=False):
        """
        Saves a grid of images to a specified file format.

        Args:
            output_file_name (str): Path to save the resulting image.
            *images (numpy.ndarray): Variable number of OpenCV images to plot (BGR by default).
            n_rows (int, optional): Number of rows in the grid. Default is 1.
            n_cols (int, optional): Number of columns in the grid. Default is 1.
            color_space (str, optional): Color space of the input images ("BGR", "RGB"). Default is "BGR".
            output_format (str, optional): Format of the output file ("TIFF", "PNG", "JPG", etc.). Default is "TIFF".
            plot_grids (bool, optional): If True, adds black dashed grid lines at 1/4, 2/4, 3/4 positions. Default is False.

        Raises:
            ValueError: If no images are provided or if n_rows * n_cols is smaller than the number of images.

        """
        if not images:
            raise ValueError("At least one image must be provided.")

        num_images = len(images)
        if n_rows * n_cols < num_images:
            raise ValueError("Grid size (n_rows * n_cols) is too small for the number of images.")

        # Create figure for grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.ravel() if n_rows * n_cols > 1 else [axes]

        # Convert all images to BGR (if needed)
        image_vec = [ColorConverter.convert_any_to_bgr(image, image_mode=color_space) for image in images]

        for i, image in enumerate(image_vec):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
            axes[i].imshow(image)
            axes[i].axis("off")

            height, width, _ = image.shape  # Get image dimensions

            # Add black border (frame)
            axes[i].plot([0, width], [0, 0], 'k-', linewidth=1)  # Top border
            axes[i].plot([0, width], [height, height], 'k-', linewidth=1)  # Bottom border
            axes[i].plot([0, 0], [0, height], 'k-', linewidth=1)  # Left border
            axes[i].plot([width, width], [0, height], 'k-', linewidth=1)  # Right border

            # Add optional grid
            if plot_grids:
                for j in range(1, 4):  # 1/4, 2/4, 3/4 divisions
                    axes[i].plot([0, width], [height * j / 4, height * j / 4], 'k--', linewidth=0.75)  # Horizontal
                    axes[i].plot([width * j / 4, width * j / 4], [0, height], 'k--', linewidth=0.75)  # Vertical

        # Remove empty subplots
        for j in range(num_images, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        # TIFF Workaround (preserves layers)
        if output_format.upper() == "TIFF":
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                plt.savefig(tmp_file.name, format="png")
                plt.close(fig)
                temp_image = Image.open(tmp_file.name)
                self.save_as_tiff(temp_image, output_file_name)
        else:
            plt.savefig(output_file_name, format=output_format.upper())
            plt.close(fig)

    def save_picture_labels(self, output_file_name, *images, pixel_counts, n_rows=1, n_cols=1, 
                            color_space="BGR", output_format="TIFF", plot_grids=False):
        """
        Saves a grid of images to a specified file format with pixel count labels below each image.

        Parameters
        ----------
        output_file_name : str
            Path to save the resulting image.
        *images : numpy.ndarray
            Variable number of OpenCV images to plot (BGR by default).
        pixel_counts : list of int
            List of precomputed total non-masked pixels for each image. Must match the number of images.
        n_rows : int, optional
            Number of rows in the grid. Default is 1.
        n_cols : int, optional
            Number of columns in the grid. Default is 1.
        color_space : str, optional
            Color space of the input images ("BGR", "RGB"). Default is "BGR".
        output_format : str, optional
            Format of the output file ("TIFF", "PNG", "JPG", etc.). Default is "TIFF".
        plot_grids : bool, optional
            If True, adds black dashed grid lines at 1/4, 2/4, 3/4 positions. Default is False.

        Raises
        ------
        ValueError
            If the number of images does not match the number of pixel counts.
            If no images are provided or if n_rows * n_cols is smaller than the number of images.
        """
        if not images:
            raise ValueError("At least one image must be provided.")

        num_images = len(images)
        
        if len(pixel_counts) != num_images:
            raise ValueError("Number of pixel counts must match the number of images.")

        if n_rows * n_cols < num_images:
            raise ValueError("Grid size (n_rows * n_cols) is too small for the number of images.")

        # Create figure for grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.ravel() if n_rows * n_cols > 1 else [axes]

        # Convert all images to BGR (if needed)
        image_vec = [ColorConverter.convert_any_to_bgr(image, image_mode=color_space) for image in images]

        for i, (image, pixel_count) in enumerate(zip(image_vec, pixel_counts)):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
            axes[i].imshow(image)
            axes[i].axis("off")

            height, width, _ = image.shape  # Get image dimensions

            # Add black border (frame)
            axes[i].plot([0, width], [0, 0], 'k-', linewidth=1)  # Top border
            axes[i].plot([0, width], [height, height], 'k-', linewidth=1)  # Bottom border
            axes[i].plot([0, 0], [0, height], 'k-', linewidth=1)  # Left border
            axes[i].plot([width, width], [0, height], 'k-', linewidth=1)  # Right border

            # Add optional grid
            if plot_grids:
                for j in range(1, 4):  # 1/4, 2/4, 3/4 divisions
                    axes[i].plot([0, width], [height * j / 4, height * j / 4], 'k--', linewidth=0.75)  # Horizontal
                    axes[i].plot([width * j / 4, width * j / 4], [0, height], 'k--', linewidth=0.75)  # Vertical

            # Add pixel count label below the image
            axes[i].text(width // 2, height + 10, f"Pixels: {pixel_count}", fontsize=10,
                         ha="center", va="top", color="black", fontweight="bold")

        # Remove empty subplots
        for j in range(num_images, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        # TIFF Workaround (preserves layers)
        if output_format.upper() == "TIFF":
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                plt.savefig(tmp_file.name, format="png")
                plt.close(fig)
                temp_image = Image.open(tmp_file.name)
                self.save_as_tiff(temp_image, output_file_name)
        else:
            plt.savefig(output_file_name, format=output_format.upper())
            plt.close(fig)

    def save_plot(self, image, output_file_name, fig=None, ax=None, segments=None, title=None,
                  color_space="BGR", output_format="TIFF", segment_color=(255, 0, 0), segment_width=1):
        """
        Save a Matplotlib plot with optional image boundary highlights.

        Parameters:
            image (numpy.ndarray): Image to be saved.
            output_file_name (str): Path to save the resulting plot.
            fig (matplotlib.figure.Figure, optional): Matplotlib figure object. If None, a new figure is created.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object. If None, a new axis is created.
            segments (numpy.ndarray, optional): Segmentation results to overlay.
            title (str, optional): Title of the plot. Default is None.
            color_space (str, optional): Color space of the input image. Default is "BGR".
            output_format (str, optional): Output file format. Default is "TIFF".
            segment_color (tuple, optional): RGB tuple for boundary color. Default is (255, 0, 0).
            segment_width (int, optional): Width of boundary lines in overlay. Default is 1.
        """

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Convert to BGR -> Then to RGB
        bgr_image = ColorConverter.convert_any_to_bgr(image, image_mode=color_space)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if segments is not None:
            overlay = np.copy(image)
            boundaries = find_boundaries(segments, mode='outer')
            overlay[boundaries] = segment_color

            if segment_width > 1:
                kernel = np.ones((segment_width, segment_width), np.uint8)
                boundaries_dilated = cv2.dilate(boundaries.astype(np.uint8), kernel, iterations=1)
                overlay[boundaries_dilated.astype(bool)] = segment_color
            ax.imshow(overlay)
        else:
            ax.imshow(image)

        if title:
            ax.set_title(title)
        ax.axis("off")

        if output_format.upper() == "TIFF":
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                fig.savefig(tmp_file.name, format="png")
                plt.close(fig)
                temp_image = Image.open(tmp_file.name)
                self.save_as_tiff(temp_image, output_file_name)
        else:
            fig.savefig(output_file_name, format=output_format.upper())
            plt.close(fig)

    def histogram(self, histograms, labels, output_file_name, output_format="TIFF", sigma_smooth=2):
        """
        Plots one or multiple histograms as smooth curves with transparent fill.

        Parameters
        ----------
        histograms : list of numpy.ndarray
            List of 1D histograms (intensity distributions) to plot.
        labels : list of str
            Labels for each histogram (for the legend). Must match `histograms` in length.
        output_file_name : str
            File path to save the histogram plot.
        output_format : str, optional
            Output format (TIFF, PNG, JPG, etc.). Default is "TIFF".
        sigma_smooth : int, optional
            Smoothing factor for the curves. Default is 2.
        
        Raises
        ------
        ValueError
            If `histograms` and `labels` lengths do not match.
        """
        if len(histograms) != len(labels):
            raise ValueError("Number of histograms must match the number of labels.")

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.cm.viridis(np.linspace(0, 1, len(histograms)))

        for hist, label, color in zip(histograms, labels, colors):
            smoothed_hist = gaussian_filter1d(hist, sigma=sigma_smooth)
            x_values = np.arange(len(hist))
            ax.plot(x_values, smoothed_hist, color=color, linewidth=2, label=label)
            ax.fill_between(x_values, smoothed_hist, alpha=0.3, color=color)

        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlabel("Intensity Level")
        ax.set_ylabel("Frequency")

        plt.tight_layout()
        
        if output_format.upper() == "TIFF":
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                plt.savefig(tmp_file.name, format="png")
                plt.close(fig)
                temp_image = Image.open(tmp_file.name)
                self.save_as_tiff(temp_image, output_file_name)
        else:
            plt.savefig(output_file_name, format=output_format.upper())
            plt.close(fig)

class ColorConverter:
    """
    Color Converter
    """

    @staticmethod
    def normalize(image, mode="BGR"):
        """
        Normalizes the pixel values of an image to the range [0, 1], considering the image mode.

        Parameters:
        -----------
        image : np.ndarray
            The input image in any color space.
        mode : str, optional
            The color space of the input image. Supported modes: "BGR", "RGB", "LAB", "HSV".
            Defaults to "BGR".

        Returns:
        --------
        np.ndarray
            The normalized image with values in the range [0, 1].

        Raises:
        -------
        ValueError
            If the mode is not supported or the image is not a valid ndarray.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a valid numpy ndarray.")

        if mode not in ["BGR", "RGB", "LAB", "HSV"]:
            raise ValueError(f"Unsupported mode '{mode}'. Supported modes are 'BGR', 'RGB', 'LAB', and 'HSV'.")

        if mode in ["BGR", "RGB"]:
            # Normalize assuming values are in the range [0, 255]
            normalized_image = image.astype(np.float32) / 255.0
            return image.astype(np.float32) 
        elif mode == "HSV":
            normalized_image = image.astype(np.float32)
            normalized_image[:, :, 0] /= 180.0  # Normalize H channel
            normalized_image[:, :, 1] /= 255.0  # Normalize S channel
            normalized_image[:, :, 2] /= 255.0  # Normalize V channel            
        elif mode == "LAB":
            # Normalize LAB channel values
            # L channel typically ranges from 0 to 100, while A and B range from -128 to 127
            normalized_image = image.astype(np.float32)
            normalized_image[:, :, 0] /= 100.0  # Normalize L channel
            normalized_image[:, :, 1] = (normalized_image[:, :, 1] + 128) / 255.0  # Normalize A channel
            normalized_image[:, :, 2] = (normalized_image[:, :, 2] + 128) / 255.0  # Normalize B channel
        
        return normalized_image

    @staticmethod
    def count_non_masked_pixels(image, color):
        """
        Counts the number of non-masked pixels in an OpenCV image where the mask is defined as (255,255,255).

        Parameters
        ----------
        image : numpy.ndarray
            The input image (assumed to be BGR or grayscale).

        Returns
        -------
        int
            The number of pixels that are NOT color.
        """
        return np.sum(np.any(image != color, axis=-1))

    @staticmethod
    def count_color_pixels(image, color):
        """
        Counts the number of non-masked pixels in an OpenCV image where the mask is defined as (255,255,255).

        Parameters
        ----------
        image : numpy.ndarray
            The input image (assumed to be BGR or grayscale).

        Returns
        -------
        int
            The number of pixels that are NOT color.
        """
        return np.sum(np.any(image == color, axis=-1))

    @staticmethod
    def opencv_to_real_lab(opencv_lab_pixel):
        """
        Convert an OpenCV LAB pixel (0-255, 0-255, 0-255) to real LAB (L: 0-100, A/B: -128 to +127).
        """
        l, a, b = opencv_lab_pixel
        l_real = l * 100 / 255
        a_real = a - 128
        b_real = b - 128
        return l_real, a_real, b_real

    @staticmethod
    def real_to_opencv_lab(real_lab_pixel):
        """
        Convert a real LAB pixel (L: 0-100, A/B: -128 to +127) to OpenCV LAB (0-255, 0-255, 0-255).
        """
        l, a, b = real_lab_pixel
        l_opencv = int(l * 255 / 100)
        a_opencv = int(a + 128)
        b_opencv = int(b + 128)
        return l_opencv, a_opencv, b_opencv

    @staticmethod
    def convert_rgb_to_hsv(rgb_color):
        """
        Convert an RGB color to HSV.
        """
        rgb_color = np.uint8([[rgb_color]])
        hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0][0]
        return tuple(hsv_color)

    @staticmethod
    def convert_hsv_to_rgb(hsv_color):
        """
        Convert an HSV color to RGB.
        """
        hsv_color = np.uint8([[hsv_color]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
        return tuple(rgb_color)

    @staticmethod
    def convert_rgb_to_cmyk(rgb_color):
        """
        Convert RGB color to CMYK.
        """
        r, g, b = rgb_color
        if r == 0 and g == 0 and b == 0:
            return 0, 0, 0, 1
        r_prime = r / 255
        g_prime = g / 255
        b_prime = b / 255
        k = 1 - max(r_prime, g_prime, b_prime)
        c = (1 - r_prime - k) / (1 - k) if k != 1 else 0
        m = (1 - g_prime - k) / (1 - k) if k != 1 else 0
        y = (1 - b_prime - k) / (1 - k) if k != 1 else 0
        return c, m, y, k

    @staticmethod
    def rotate_hsv(hsv_image, rotation=120):
        """
        Rotates the hue channel of an HSV image so the 180-0 breaking point is shifted to cyan/blue.

        Parameters:
        -----------
        hsv_image : np.ndarray
            The input HSV image.
        rotation : int, optional
            The rotation value to add to the hue channel (default is 120).

        Returns:
        --------
        np.ndarray
            The rotated HSV image.
        """
        hsv_rotated = hsv_image.copy()
        hsv_rotated[:, :, 0] = (hsv_rotated[:, :, 0] + rotation) % 180
        return hsv_rotated

    @staticmethod
    def convert_any_to_bgr(image, image_mode="BGR"):
        """
        Converts an input image to OpenCV's BGR format.
        
        Parameters:
        ----------
        image : np.ndarray
            Input image in an unknown color space.

        Returns:
        -------
        np.ndarray
            Image converted to BGR.
        """

        if image_mode.upper() == "BGR":
            pass
        elif image_mode.upper() == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image_mode.upper() == "GRAY":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image_mode.upper() == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif image_mode.upper() == "HLS":
            image = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
        elif image_mode.upper() == "LAB":
            image = cv2.cvtColor(image, cv2.COLOR_Lab2BGR)
        elif image_mode.upper() == "LUV":
            image = cv2.cvtColor(image, cv2.COLOR_Luv2BGR)
        elif image_mode.upper() == "XYZ":
            image = cv2.cvtColor(image, cv2.COLOR_XYZ2BGR)
        elif image_mode.upper() == "YCRCB":
            image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        else:
            try:
                if len(image.shape) == 2:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[-1] == 3:
                    # Check if it's RGB instead of BGR
                    if image[..., 0].mean() >= image[..., 2].mean():  # More blue in BGR
                        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError(f"The image '{original_image}' encoding could not be determined.")

        return image

    @staticmethod
    def apply_masked_background(image, mask, color_mode="HSV", background_color="white"):
        """
        Applies the mask to an image, replacing masked-out areas with a background color.

        Parameters
        ----------
        color_mode : str
            The color mode of the image. Currently supports 'HSV'.
        background_color : str
            The background color to apply. Must be one of:
            ['white', 'black', 'blue', 'red', 'green', 'yellow'].

        Returns
        -------
        numpy.ndarray
            The image with the mask applied and the background set.
        """
        # Background color mapping for HSV (extend this if needed)
        BG_COLORS = {
            "white": {"HSV": (0, 0, 255)},
            "black": {"HSV": (0, 0, 0)},
            "blue": {"HSV": (120, 255, 255)},
            "red": {"HSV": (0, 255, 255)},
            "green": {"HSV": (60, 255, 255)},
            "yellow": {"HSV": (30, 255, 255)}
        }

        # Validate background color
        if background_color not in BG_COLORS:
            raise ValueError(f"Invalid background color. Choose from {list(BG_COLORS.keys())}")

        # Restrict to HSV for now (can extend later)
        if color_mode != "HSV":
            raise ValueError(f"Invalid color mode '{color_mode}'. Only 'HSV' is supported.")

        # Get background color and create the background image
        bg_color = BG_COLORS[background_color][color_mode]
        bg_image = np.full(image.shape, bg_color, dtype=image.dtype)

        # Apply mask to retain selected parts of the original image
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Apply inverted mask to fill in background
        inverted_mask = cv2.bitwise_not(mask)
        bg_masked = cv2.bitwise_and(bg_image, bg_image, mask=inverted_mask)

        # Combine masked image with background
        final_image = cv2.add(masked_image, bg_masked)

        return final_image

    @staticmethod
    def bgr_to_munsell(image_bgr):
        """
        Converts an OpenCV BGR image to the Munsell color system.

        Parameters:
        -----------
        image_bgr : np.ndarray
            The input image in BGR format (as loaded by OpenCV).

        Returns:
        --------
        munsell_image : list of list of str
            A 2D list where each element is the Munsell notation corresponding to the pixel in the input image.
        """
        # Check if the input is a valid BGR image
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.shape[2] != 3:
            raise ValueError("Input must be a valid BGR image with 3 channels.")

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Normalize the RGB values to the range [0, 1]
        image_rgb_normalized = image_rgb / 255.0

        # Prepare an empty list to store Munsell notations
        munsell_image = []

        # Iterate over each pixel in the image
        for row in image_rgb_normalized:
            munsell_row = []
            for pixel in row:
                # Convert the RGB pixel to XYZ
                xyz = colour.sRGB_to_XYZ(pixel)

                # Convert XYZ to xyY
                xyy = colour.XYZ_to_xyY(xyz)

                # Convert xyY to Munsell notation
                munsell = colour.xyY_to_munsell_colour(xyy)

                # Append the Munsell notation to the row list
                munsell_row.append(munsell)

            # Append the row to the Munsell image list
            munsell_image.append(munsell_row)

        return munsell_image


class TripleInterval:
    """
    Represents a modifiable 3D interval where intervals can be added, removed, and split dynamically.
    """

    def __init__(self, start, end):
        """
        Initializes a 3D interval.

        Parameters
        ----------
        start : tuple (float, float, float)
            The (x, y, z) start values of the interval.
        end : tuple (float, float, float)
            The (x, y, z) end values of the interval.
        """
        self.intervals = [(start, end)]  # Store multiple non-overlapping sub-intervals

    def contains(self, point):
        """
        Checks if a point is inside any of the stored sub-intervals.

        Parameters
        ----------
        point : tuple (float, float, float)
            The (x, y, z) value to check.

        Returns
        -------
        bool
            True if the point is inside, False otherwise.
        """
        for (s, e) in self.intervals:
            if all(s[i] <= point[i] <= e[i] for i in range(3)):
                return True
        return False

    def remove_interval(self, start, end):
        """
        Removes a sub-interval by splitting the existing intervals.

        Parameters
        ----------
        start : tuple (float, float, float)
            The (x, y, z) start values of the interval to remove.
        end : tuple (float, float, float)
            The (x, y, z) end values of the interval to remove.
        """
        new_intervals = []
        for (s, e) in self.intervals:
            # Check if the removal interval overlaps with the current interval
            if all(start[i] > e[i] or end[i] < s[i] for i in range(3)):
                new_intervals.append((s, e))  # No overlap, keep the interval
            else:
                # Split into possible remaining intervals
                for i in range(3):
                    if start[i] > s[i]:  # Left-side remains
                        new_s = list(s)
                        new_e = list(e)
                        new_e[i] = start[i]
                        new_intervals.append((tuple(new_s), tuple(new_e)))
                    if end[i] < e[i]:  # Right-side remains
                        new_s = list(s)
                        new_e = list(e)
                        new_s[i] = end[i]
                        new_intervals.append((tuple(new_s), tuple(new_e)))

        self.intervals = new_intervals  # Update stored intervals

    def add_interval(self, start, end):
        """
        Adds a new interval to the stored list while merging overlapping intervals.

        Parameters
        ----------
        start : tuple (float, float, float)
            The (x, y, z) start values of the interval to add.
        end : tuple (float, float, float)
            The (x, y, z) end values of the interval to add.
        """
        self.intervals.append((start, end))
        # Merge overlapping intervals
        self._merge_intervals()

    def _merge_intervals(self):
        """
        Merges overlapping intervals to ensure the list remains minimal.
        """
        merged = []
        for (s, e) in sorted(self.intervals):
            if not merged or any(e[i] < merged[-1][0][i] or s[i] > merged[-1][1][i] for i in range(3)):
                merged.append((s, e))  # No overlap, add new interval
            else:
                # Merge with previous interval
                prev_s, prev_e = merged.pop()
                new_s = tuple(min(prev_s[i], s[i]) for i in range(3))
                new_e = tuple(max(prev_e[i], e[i]) for i in range(3))
                merged.append((new_s, new_e))

        self.intervals = merged

    def __eq__(self, other):
        """
        Checks if two TripleInterval objects are equal.

        Parameters
        ----------
        other : TripleInterval
            The other TripleInterval object to compare.

        Returns
        -------
        bool
            True if both objects have the same intervals, False otherwise.
        """
        if not isinstance(other, TripleInterval):
            return False

        # Use frozenset to ignore order of intervals
        return frozenset(self.intervals) == frozenset(other.intervals)

    def __repr__(self):
        """
        String representation of the stored intervals.
        """
        return f"TripleInterval({self.intervals})"


















