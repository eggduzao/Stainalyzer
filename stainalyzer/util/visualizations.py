
"""
Visualizations
"""

############################################################################################################
### Import
############################################################################################################

import io
import os
import cv2
import tempfile
import numpy as np
from math import ceil, sqrt
from PIL import Image, ImageCms
import matplotlib.pyplot as plt

from .utils import ColorName, ColorConverter, PlottingUtils

############################################################################################################
### Constants
############################################################################################################

# Constants
Image.MAX_IMAGE_PIXELS = 400_000_000  # Adjust this as needed
SEED = 1987
np.random.seed(SEED)

############################################################################################################
### Classes
############################################################################################################

class Visualizations:
    """
    Visualizations TODO DESCRIPTION
    """

    def __init__(self, color_name_path=None):
        """
        Initialize the PlottingUtils class.

        Parameters:
            srgb_profile (object, optional): A color profile for image processing. 
                                             Defaults to an sRGB profile if not provided.
        """

        # Checking attributes
        if color_name_path is None:
            raise ValueError("color_name_path must be provided.")
        else:
            self.color_name_path = color_name_path

        self.color_name_dict = ColorName(self.color_name_path)
        self.plotting_utils = PlottingUtils()

    def save_clahe_image(self, clahe_image, output_file_name, color_space="LAB", output_format="TIFF"):
        """
        Save a visualization of SLIC superpixel segmentation with boundaries highlighted.

        This function uses the `save_picture` utility to save the CLAHE-normalized image with
        optional specifications for color space and file format.

        Parameters:
            clahe_image (numpy.ndarray): The CLAHE-normalized image to save, typically in LAB.
            output_file_name (str): The file path to save the image.
            color_space (str, optional): The color space of the input image ("LAB" by default).
                                         Options include "LAB", "HSV", "BGR", and "RGB".
            output_format (str, optional): The format in which to save the image. Default is "TIFF".

        Returns:
            None
        """

        # Save the visualization using the save_plot function
        self.plotting_utils.save_picture(output_file_name,
                                         clahe_image,
                                         color_space=color_space,
                                         output_format=output_format
                                         )

    def save_superpixel_visualization(self, image, segments, output_file_name, color_space="BGR", boundary_color=(255,0,0), boundary_width=1, title="SLIC Segmentation"):
        """
        Save a visualization of SLIC superpixel segmentation with boundaries highlighted.

        Parameters:
            image (numpy.ndarray): Input image for printing (BGR, RGB, LAB, etc...).
            segments (numpy.ndarray): The segmentation result from SLIC.
            output_file_name (str): Path to save the output image.
            boundary_color (str, optional): Color for the boundaries. Default is "red".
            title (str, optional): Title for the visualization. Default is "SLIC Segmentation".

        Returns:
            None
        """
        # Ensure that the segmentation array matches the image dimensions
        if image.shape[:2] != segments.shape:
            raise ValueError("Segmentation shape must match the spatial dimensions of the input image.")

        # Save the visualization using the save_plot function
        self.plotting_utils.save_plot(image,
                                      output_file_name,
                                      segments=segments,
                                      title=title,
                                      color_space=color_space,
                                      segment_color=boundary_color,
                                      segment_width=boundary_width
                                      )

    def save_quantized_image(self, quantized_image, output_file_name, color_space="LAB", output_format="TIFF"):
        """
        Save a quantized image to a specified file for visualization.

        This function uses the `save_picture` utility to save the quantized image with optional
        specifications for color space and file format.

        Parameters:
            quantized_image (numpy.ndarray): The quantized image to save, typically in LAB color space.
            output_file_name (str): The file path to save the image.
            color_space (str, optional): The color space of the input image ("LAB" by default).
                                         Options include "LAB", "HSV", "BGR", and "RGB".
            output_format (str, optional): The format in which to save the image. Default is "TIFF".

        Notes:
            - The function assumes the input image is pre-processed (e.g., CLAHE-normalized).
            - The `save_picture` utility handles conversions to ensure consistent visualization.
        """
        self.plotting_utils.save_picture(output_file_name,
                                         quantized_image,
                                         n_rows=1,
                                         n_cols=1,
                                         color_space=color_space,
                                         output_format=output_format
                                         )

    def plot_image_cut_by_cluster(self, image, labels, centroids, output_file_name, output_path, prefix="cluster_", 
                                  color_space="LAB", output_format="TIFF"):
        """
        Plot and save cluster-separated images from an input image using cluster indices.

        This function generates two outputs:
        1. A grid of cluster-separated images saved to a single file.
        2. Individual cluster-separated images saved as separate files, one per cluster.

        Parameters:
            image (numpy.ndarray): The input image (can be real or quantized). Shape: (height, width, channels).
            labels (numpy.ndarray): Cluster labels for each pixel. Shape: (height, width).
            centroids (numpy.ndarray): Cluster centroids in LAB space. Shape: (num_clusters, 3).
            output_file_name (str): File path to save the grid of cluster-separated images.
            output_path (str): Directory path to save individual cluster-separated images.
            prefix (str, optional): Prefix for individual cluster image filenames. Default is "cluster_".
            color_space (str, optional): Color space of the input images ("LAB" by default). Options: "LAB", "RGB".
            output_format (str, optional): Format to save the images ("TIFF", "PNG", etc.). Default is "TIFF".

        Raises:
            ValueError: If `output_path` is not a valid directory.

        Notes:
            - Cluster IDs are based on the provided labels array.
            - The output grid provides a comprehensive overview, while individual files allow focused analysis.
        """

        # Ensure the output path exists
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Output path '{output_path}' is not a valid directory or cannot be created. Original error: {e}")

        # Calculate number of clusters
        num_clusters = len(centroids)

        # Initialize variables
        n_rows = int(ceil(sqrt(num_clusters)))
        n_cols = int(ceil(num_clusters / n_rows))

        # Create masks for each cluster and generate masked images
        cluster_images = []
        for cluster_id in range(num_clusters):

            # Create a mask for the current cluster
            mask = labels == cluster_id

            # Create a blank LAB image with the background set to LAB black
            cluster_image = np.full_like(image, [0, 128, 128])  # LAB black background
            cluster_image[mask] = image[mask]

            # Add to the grid visualization list
            cluster_images.append(cluster_image)

            # Save the individual cluster image
            cluster_file_name = os.path.join(output_path, f"{prefix}{cluster_id}.tiff")
            self.plotting_utils.save_picture(cluster_file_name,
                                             cluster_image,
                                             n_rows=1,
                                             n_cols=1,
                                             color_space=color_space
                                             )

        # Save the grid of images
        self.plotting_utils.save_picture(output_file_name,
                                         *cluster_images,
                                         n_rows=n_rows,
                                         n_cols=n_cols,
                                         color_space=color_space
                                         )

    def plot_cluster_colors(self, cluster_centers, pixel_counts, output_file, color_space="LAB"):
        """
        Plot a grid displaying the color of each cluster from K-means with LAB, RGB, HSV values,
        along with the total number of pixels and the closest color name for each cluster.

        Parameters:
            cluster_centers (numpy.ndarray): Cluster center colors in LAB format. Shape: (num_clusters, 3).
            pixel_counts (numpy.ndarray): Number of pixels per cluster. Shape: (num_clusters,).
            hexadecimal_to_name (dict): Dictionary to map hexadecimal color codes to color names.
            output_file (str): Path to save the resulting plot.
            color_space (str, optional): Color space of the cluster centers. Default is "LAB".
        """

        # Convert LAB cluster centers to RGB for display
        if color_space.upper() == "LAB":
            rgb_centers = cv2.cvtColor(cluster_centers[np.newaxis, :, :].astype("uint8"), cv2.COLOR_LAB2RGB)[0]
        else:
            raise ValueError(f"Unsupported color space: {color_space}")

        # Calculate grid dimensions
        num_clusters = len(cluster_centers)
        grid_size = ceil(num_clusters**0.5)
        n_rows, n_cols = grid_size, grid_size

        # Create a blank canvas for display
        box_size = 100
        canvas = np.ones((n_rows * box_size, n_cols * box_size, 3), dtype=np.uint8) * 255

        # Create a Matplotlib figure
        fig, ax = plt.subplots(figsize=(n_cols * 2, n_rows * 2))
        ax.imshow(canvas)
        ax.axis("off")

        # Iterate over clusters
        for idx, (lab_color, rgb_color) in enumerate(zip(cluster_centers, rgb_centers)):

            # Extract LAB and RGB values
            l, a, b = lab_color
            r, g, b_rgb = rgb_color

            # Convert RGB to HSV
            h, s, v = ColorConverter.convert_rgb_to_hsv((r, g, b_rgb))

            # Determine the closest color name
            color_name = self.color_name_dict.closest_color_name((r, g, b_rgb))

            # Total pixels in the cluster
            total_pixels = pixel_counts[idx]

            # Calculate grid position
            row, col = divmod(idx, grid_size)
            x_start, y_start = col * box_size, row * box_size

            # Fill the grid square with the RGB color
            canvas[y_start:y_start + box_size, x_start:x_start + box_size] = rgb_color

            # Add text to the center
            text = (
                f"{color_name}\n"
                f"Pixels: {total_pixels}\n"
                f"RGB: ({r}, {g}, {b_rgb})\n"
                f"HSV: ({h:.1f}, {s:.1f}, {v:.1f})\n"
                f"LAB: ({l:.1f}, {a:.1f}, {b:.1f})"
            )
            ax.text(
                x_start + box_size // 2,
                y_start + box_size // 2,
                text,
                fontsize=6,
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                transform=ax.transData
            )

        # Save the grid with annotations
        self.plotting_utils.save_plot(canvas,
                                      output_file,
                                      fig=fig,
                                      ax=ax,
                                      title="Cluster Colors with Annotations",
                                      color_space="RGB"
                                      )

    # Function to plot a GaussianDistribution in 2D
    def plot_gaussian_distribution_2d(gaussian_distribution, mode="BGR"):
        """
        Plots a GaussianDistribution in 2D with hue on the x-axis and Saturation x Value on the y-axis.

        Parameters:
        -----------
        gaussian_distribution : GaussianDistribution
            An instance of the GaussianDistribution class with normalized color values.
        mode : str, optional
            The color space mode of the GaussianDistribution (default is "BGR").

        Returns:
        --------
        None
            Displays the 2D plot.
        """
        # Extract mean and covariance from the GaussianDistribution
        mean, covariance = gaussian_distribution.get_mean_and_covariance()

        # Convert the mean to HSV if not already in HSV
        if mode == "BGR":
            mean_hsv = cv2.cvtColor(np.uint8([[mean]]), cv2.COLOR_BGR2HSV)[0][0]
        elif mode == "RGB":
            mean_hsv = cv2.cvtColor(np.uint8([[mean]]), cv2.COLOR_RGB2HSV)[0][0]
        elif mode == "LAB":
            mean_bgr = cv2.cvtColor(np.uint8([[mean]]), cv2.COLOR_LAB2BGR)[0][0]
            mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        elif mode == "Munsell":
            raise NotImplementedError("Munsell to HSV conversion is not implemented.")
        elif mode == "HSV2":
            mean_hsv = mean  # Assume HSV2 is already rotated HSV
        else:
            mean_hsv = mean  # Assume input is already in HSV

        # Rotate the HSV if mode is not HSV2
        if mode != "HSV2":
            mean_hsv = rotate_hsv(mean_hsv)

        # Generate synthetic points based on the Gaussian distribution
        points = np.random.multivariate_normal(mean[:2], covariance[:2, :2], 5000)

        # Extract hue and calculate Saturation x Value
        hues = points[:, 0] % 180
        saturation_value = points[:, 1] * points[:, 2]

        # Normalize hue and saturation_value for plotting
        hues = normalize(hues)
        saturation_value = normalize(saturation_value)

        # Create a scatter plot with hue on the x-axis and Saturation x Value on the y-axis
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(hues, saturation_value, c=hues, cmap="hsv", alpha=0.5)
        plt.colorbar(scatter, label="Hue (rotated)")
        plt.xlabel("Hue")
        plt.ylabel("Saturation x Value")
        plt.title("2D Gaussian Distribution Plot in HSV Space")
        plt.show()

    """
    def plot_annotated_clusters_in_quantized_image(image, quantized_image, centroids, hexadecimal_to_name, output_file_name, color_space="BGR", quantized_color_space="LAB", output_format="TIFF"):
        
        Plot clusters in a quantized image with annotations showing the color name of each cluster.

        Parameters:
            image (numpy.ndarray): The original image.
            quantized_image (numpy.ndarray): The quantized image with cluster labels.
            centroids (numpy.ndarray): A list of centroid RGB values (one for each cluster).
            hexadecimal_to_name (dict): A dictionary mapping hex color codes to color names.
            output_file_name (str): Path to save the resulting image.
            color_space (str, optional): Color space of the input images ("BGR", "RGB"). Default is "BGR".
            output_format (str, optional): Format of the output file ("TIFF", "PNG", "JPG", etc.). Default is "TIFF".
        

        # Convert image to RGB if necessary
        if color_space.upper() == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if quantized_color_space.upper() == "LAB":
            quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2RGB)
            centroids = cv2.cvtColor(centroids[np.newaxis, :, :].astype("uint8"), cv2.COLOR_LAB2RGB)[0]

        # Identify unique clusters
        unique_clusters = np.unique(quantized_image.reshape(-1, 3), axis=0)
        num_clusters = len(unique_clusters)

        # Calculate grid size
        n_rows = int(ceil(sqrt(num_clusters)))
        n_cols = int(ceil(num_clusters / n_rows))

        # Create a new figure for the grid
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
        axs = axs.flatten()

        for idx, cluster_color in enumerate(unique_clusters):
            # Create a mask for the current cluster
            mask = (quantized_image == cluster_color).all(axis=-1)

            # Create a blank image with only the cluster pixels
            cluster_image = np.zeros_like(image)
            cluster_image[mask] = image[mask]

            # Plot the cluster image
            axs[idx].imshow(cluster_image)
            axs[idx].axis("off")

            # Get the centroid RGB value and color name
            centroid_rgb = centroids[idx]
            color_name = closest_color_name(tuple(centroid_rgb), hexadecimal_to_name)

            # Add the annotation above the grid
            axs[idx].set_title(color_name, fontsize=8, pad=10)

        # Remove any unused subplots
        for ax in axs[len(unique_clusters):]:
            ax.axis("off")

        # Save the annotated plot, passing both fig and ax
        PlotUtils.save_plot(None, output_file_name, fig=fig, ax=axs, color_space="RGB", output_format=output_format)
    """

"""
# Calculate Color Distributions
def calculate_color_distributions(rgb_image, hsv_image, lab_image, quantized_image, cluster_centers, pixel_counts, hexadecimal_to_name, output_file):
    "" "
    Calculate and write color distributions for clusters in a hyper-segmented image.

    This function computes the mean and standard deviation of RGB and HSV channels
    for both real and quantized images, based on cluster centroids. It also writes
    the results to a provided output file in a tab-separated format.

    Parameters:
        rgb_image (numpy.ndarray): The real image in RGB format.
        hsv_image (numpy.ndarray): The real image in HSV format.
        lab_image (numpy.ndarray): The real image in LAB format.
        quantized_image (numpy.ndarray): The quantized image in LAB format (indexed by clusters).
        cluster_centers (numpy.ndarray): A list of centroid RGB values (one for each cluster).
        pixel_counts (list): A list of pixel counts for each cluster.
        hexadecimal_to_name (dict): A dictionary mapping hex color codes to color names.
        output_file (file object): An open file object to write the output.

    Returns:
        tuple: A tuple containing four lists of distributions:
            - Real RGB distributions (list of RGBColorDistribution objects).
            - Real HSV distributions (list of HSVColorDistribution objects).
            - Quantized RGB distributions (list of RGBColorDistribution objects).
            - Quantized HSV distributions (list of HSVColorDistribution objects).
    "" "
    # Containers for color distributions
    real_rgb_distributions = []
    real_hsv_distributions = []
    real_lab_distributions = []
    quantized_rgb_distributions = []
    quantized_hsv_distributions = []
    quantized_lab_distributions = []

    # Convert quantized image from LAB to RGB
    quantized_image_rgb = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2RGB)

    # Convert quantized image from LAB to HSV
    quantized_image_hsv = cv2.cvtColor(quantized_image_rgb, cv2.COLOR_RGB2HSV)

    # Convert LAB image to LAB
    quantized_image_lab = quantized_image

    # Convert LAB cluster centers to RGB
    rgb_centers = cv2.cvtColor(cluster_centers[np.newaxis, :, :].astype("uint8"), cv2.COLOR_LAB2RGB)[0]

    # Convert LAB cluster centers to HSV
    hsv_centers = cv2.cvtColor(rgb_centers[np.newaxis, :, :].astype("uint8"), cv2.COLOR_RGB2HSV)[0]

    # LAB Centroids
    lab_centers = cluster_centers

    # Iterate through centroids
    for idx, (rgb_color, hsv_color, lab_color) in enumerate(zip(rgb_centers, hsv_centers, lab_centers)):

        # Extract RGB pixel values for the current cluster in RGB images
        rgb_cluster_mask = np.all(quantized_image_rgb == rgb_color, axis=-1)
        cluster_pixels_rgb = rgb_image[rgb_cluster_mask]
        cluster_pixels_quantized_rgb = quantized_image_rgb[rgb_cluster_mask]

        # Extract HSV pixel values for the current cluster in HSV images
        hsv_cluster_mask = np.all(quantized_image_hsv == hsv_color, axis=-1)
        cluster_pixels_hsv = hsv_image[hsv_cluster_mask]
        cluster_pixels_quantized_hsv = quantized_image_hsv[hsv_cluster_mask]

        # Extract LAB pixel values for the current cluster in LAB images
        lab_cluster_mask = np.all(quantized_image_lab == lab_color, axis=-1)
        cluster_pixels_lab = lab_image[lab_cluster_mask]
        cluster_pixels_quantized_lab = quantized_image_lab[lab_cluster_mask]
        
        # Handle empty clusters
        if cluster_pixels_rgb.size == 0:
            print(f"Warning: No pixels found for centroid {idx}. Skipping.")
            continue

        # Calculate color name
        color_name = closest_color_name(tuple(rgb_color), hexadecimal_to_name)

        # Calculate Real RGB Distributions
        real_rgb_mean = np.mean(cluster_pixels_rgb, axis=0)
        real_rgb_std = np.std(cluster_pixels_rgb, axis=0)
        real_rgb_distributions.append(RGBColorDistribution(color_name, *real_rgb_mean, *real_rgb_std))

        # Calculate Real HSV Distributions
        real_hsv_mean = np.mean(cluster_pixels_hsv, axis=0)
        real_hsv_std = np.std(cluster_pixels_hsv, axis=0)
        real_hsv_distributions.append(HSVColorDistribution(color_name, *real_hsv_mean, *real_hsv_std))

        # Calculate Real LAB Distributions
        real_lab_mean = np.mean(cluster_pixels_lab, axis=0)
        real_lab_std = np.std(cluster_pixels_lab, axis=0)
        real_lab_distributions.append(LABColorDistribution(color_name, *real_lab_mean, *real_lab_std))

        # Calculate Quantized RGB Distributions
        quantized_rgb_mean = np.mean(cluster_pixels_quantized_rgb, axis=0)
        quantized_rgb_std = np.zeros((3,), dtype=np.float64)
        quantized_rgb_distributions.append(RGBColorDistribution(color_name, *quantized_rgb_mean, *quantized_rgb_std))

        # Calculate Quantized HSV Distributions
        quantized_hsv_mean = np.mean(cluster_pixels_quantized_hsv, axis=0)
        quantized_hsv_std = np.zeros((3,), dtype=np.float64)
        quantized_hsv_distributions.append(HSVColorDistribution(color_name, *quantized_hsv_mean, *quantized_hsv_std))

        # Calculate Quantized LAB Distributions
        quantized_lab_mean = np.mean(cluster_pixels_quantized_lab, axis=0)
        quantized_lab_std = np.zeros((3,), dtype=np.float64)
        quantized_lab_distributions.append(LABColorDistribution(color_name, *quantized_lab_mean, *quantized_lab_std))

        # --- Write Results to Output File ---
        output_file.write("\t".join(map(str, [
            color_name,  # Color name
            pixel_counts[idx] if idx < len(pixel_counts) else 0,  # Total pixels in cluster
            *rgb_color,  # Centroid RGB
            *hsv_color,  # Centroid HSV
            *lab_color,  # Centroid LAB
            *real_rgb_mean, *real_rgb_std,  # Real RGB stats
            *real_hsv_mean, *real_hsv_std,  # Real HSV stats
            *real_lab_mean, *real_lab_std,  # Real HSV stats
            *quantized_rgb_mean, *quantized_rgb_std,  # Quantized RGB stats
            *quantized_hsv_mean, *quantized_hsv_std,  # Quantized HSV stats
            *quantized_lab_mean, *quantized_lab_std  # Quantized HSV stats
        ])) + "\n")

    # Return all distributions as a tuple
    return real_rgb_distributions, real_hsv_distributions, real_lab_distributions, quantized_rgb_distributions, quantized_hsv_distributions, quantized_lab_distributions

"""

