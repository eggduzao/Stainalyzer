
"""
Script for processing images:
1. Read image in RGB, HSV and LAB.
2. Remove all black parts.
3. Calculate dynamic threshold.
4. Perform superpixel segmentation (SLIC).
5. Perform k-means clustering on segmented image.
6. Extracting HSV distributions from cluster colors.
7. Create useful plots and provide results as txt.
"""

##################################
### Import
##################################

# Import required libraries
import io
import os
import cv2
import math
import struct
import tempfile
import numpy as np
import seaborn as sns
from math import ceil
from time import perf_counter
from PIL import Image, ImageCms
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries

##################################
### Constants
##################################

# Constants
Image.MAX_IMAGE_PIXELS = 400_000_000  # Adjust this as needed
SEED = 1987
np.random.seed(SEED)

##################################
### Classes
##################################

class HSVColorDistribution:
    """
    A class to represent an HSV color distribution.
    """
    def __init__(self, name, hue_mean, hue_std, saturation_mean, saturation_std, value_mean, value_std):
        self.name = name
        self.hue_mean = hue_mean
        self.hue_std = hue_std
        self.saturation_mean = saturation_mean
        self.saturation_std = saturation_std
        self.value_mean = value_mean
        self.value_std = value_std

    def as_array(self):
        return np.array([
            self.hue_mean, self.hue_std,
            self.saturation_mean, self.saturation_std,
            self.value_mean, self.value_std
        ])

class RGBColorDistribution:
    """
    A class to represent an RGB color distribution.
    """
    def __init__(self, name, red_mean, red_std, green_mean, green_std, blue_mean, blue_std):
        self.name = name
        self.red_mean = red_mean
        self.red_std = red_std
        self.green_mean = green_mean
        self.green_std = green_std
        self.blue_mean = blue_mean
        self.blue_std = blue_std

    def as_array(self):
        return np.array([
            self.red_mean, self.red_std,
            self.green_mean, self.green_std,
            self.blue_mean, self.blue_std
        ])

class LABColorDistribution:
    """
    A class to represent an LAB color distribution.
    """
    def __init__(self, name, l_mean, l_std, a_mean, a_std, b_mean, b_std):
        self.name = name
        self.l_mean = l_mean
        self.l_std = l_std
        self.a_mean = a_mean
        self.a_std = a_std
        self.b_mean = b_mean
        self.b_std = b_std

    def as_array(self):
        return np.array([
            self.lightness_mean, self.lightness_std,
            self.a_mean, self.a_std,
            self.b_mean, self.b_std
        ])

##################################
### Centralized Color Utilities
##################################

# Load File with Hexadecimal Color Names from Community
def load_hexadecimal_to_name(file_path):
    """
    Load the hexadecimal-to-name mapping from a community color-vote csv-table file.
    Args:
        file_path (str): Path to the color names file.
    Returns:
        dict: A dictionary mapping hexadecimal color codes to their names.
    """
    hexadecimal_to_name = {}
    with open(file_path, 'r') as f:
        f.readline()
        for line in f:
            # Split line into columns
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            hex_color, name = parts[0], parts[1]
            hexadecimal_to_name[hex_color] = name
    return hexadecimal_to_name

# Convert RGB to HSV
def convert_rgb_to_hsv(rgb_color):
    """
    Convert an RGB color to HSV.
    """
    rgb_color = np.uint8([[rgb_color]])
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0][0]
    return tuple(hsv_color)

# Convert HSV to RGB
def convert_hsv_to_rgb(hsv_color):
    """
    Convert an HSV color to RGB.
    """
    hsv_color = np.uint8([[hsv_color]])
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
    return tuple(rgb_color)

# Convert RGB to CMYK
def rgb_to_cmyk(rgb_color):
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

def calculate_color_distance(color1, color2, mode='manhattan'):
    """
    Calculate the distance between two colors.
    """
    if mode == 'manhattan':
        return sum(abs(c1 - c2) for c1, c2 in zip(color1, color2))
    elif mode == 'euclidean':
        return sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)) ** 0.5
    else:
        raise ValueError("Unsupported mode. Use 'manhattan' or 'euclidean'.")

def closest_color_name(rgb_color, hexadecimal_to_name):
    """
    Find the closest named color to the given RGB color.
    Args:
        rgb_color (tuple): A tuple representing the RGB color (R, G, B).
        hexadecimal_to_name (dict): Dictionary of hex-to-name mappings.
    Returns:
        str: The name of the closest color.
    """

    # Convert RGB to lowercase hexadecimal
    hex_color = f"{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"
    
    # Exact match
    if hex_color in hexadecimal_to_name:
        return hexadecimal_to_name[hex_color]

    # Zigzag search for the closest color
    min_distance = float('inf')
    closest_name = None
    for hex_key, name in hexadecimal_to_name.items():

        # Convert hex_key to RGB (float otherwise warning)
        r = float(int(hex_key[0:2], 16))
        g = float(int(hex_key[2:4], 16))
        b = float(int(hex_key[4:6], 16))

        # Calculate Manhattan distance
        distance = calculate_color_distance((r, g, b), rgb_color)
        if distance < min_distance:
            min_distance = distance
            closest_name = name

    return closest_name

# Save as TIFF
def save_as_tiff(image, file_path):
    """
    General function to save an image as a TIFF with a standard sRGB ICC profile.
    """

    # Save as TIFF using PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image = ImageCms.profileToProfile(pil_image, srgb_profile, srgb_profile, outputMode="RGB")
    pil_image.save(file_path, format="TIFF")

##################################
### Printing Utilities
##################################

# Save picture(s)
def save_picture(output_file_name, *images, n_rows=1, n_cols=1, color_space="BGR", output_format="TIFF"):
    """
    Save a grid of images to a specified file format.

    Parameters:
        output_file_name (str): Path to save the resulting image.
        *images (numpy.ndarray): Variable number of OpenCV images to plot (BGR by default).
        n_rows (int): Number of rows in the grid. Default is 1.
        n_cols (int): Number of columns in the grid. Default is 1.
        color_space (str): Color space of the input images ("BGR", "RGB"). Default is "BGR".
        output_format (str): Format of the output file ("TIFF", "PNG", "JPG", etc.). Default is "TIFF".

    Raises:
        ValueError: If no images are provided or if n_rows * n_cols < number of images.
    """
    if not images:
        raise ValueError("At least one image must be provided.")
    
    num_images = len(images)
    if n_rows * n_cols < num_images:
        raise ValueError("Grid size (n_rows * n_cols) is too small for the number of images.")

    # Create a figure for the grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.ravel() if n_rows * n_cols > 1 else [axes]

    for i, image in enumerate(images):
        if color_space.upper() == "RGB":
            # Already in RGB
            pass
        elif color_space.upper() == "BGR":
            # Convert BGR to RGB for correct display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space.upper() == "HSV":
            # Convert HSV to RGB for correct display
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        elif color_space.upper() == "LAB":
            # Convert LAB to RGB for correct display
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        else:
            raise ValueError("A correct color space must be provided.")
        axes[i].imshow(image)
        axes[i].axis("off")

    # Turn off unused axes
    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # Save as a standardized TIFF if selected
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        plt.savefig(tmp_file.name, format="png")
        plt.close()

        temp_image = Image.open(tmp_file.name)
        if output_format.upper() == "TIFF":
            temp_image = ImageCms.profileToProfile(temp_image, srgb_profile, srgb_profile, outputMode="RGB")
        temp_image.save(output_file_name, format=output_format.upper())

# Save Plot
def save_plot(image, output_file_name, fig=None, ax=None, segments=None, title=None, color_space="BGR", output_format="TIFF"):
    """
    Save a Matplotlib plot with optional image boundary highlights to a specified file format.

    Parameters:
        image (numpy.ndarray): Image to be saved.
        output_file_name (str): Path to save the resulting plot.
        boundary_color (tuple, list, or str, optional): RGB color to highlight boundaries (e.g., (255, 0, 0) for red). Default is None.
        fig (matplotlib.figure.Figure, optional): Matplotlib figure object. If None, a new figure will be created.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis object. If None, a new axis will be created.
        title (str, optional): Title of the plot. Default is None (no title).
        output_format (str, optional): Format of the output file ("TIFF", "PNG", "JPG", etc.). Default is "TIFF".
        color_space (str, optional): Color space of the input image ("BGR", "RGB", "HSV", "LAB"). Default is "BGR".

    Raises:
        ValueError: If an invalid color space or output format is provided.

    Notes:
        If `boundary_color` is provided, boundaries will be overlaid on the image.
        The default output format ("TIFF") ensures the saved image uses an sRGB ICC profile for color consistency.
    """

    # Create a new figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if image is not None:
        # Convert image to RGB if necessary
        if color_space.upper() == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space.upper() == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        elif color_space.upper() == "LAB":
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        elif color_space.upper() != "RGB":
            raise ValueError(f"Unsupported color space: {color_space}")

    if fig is None and ax is None and image is None:
        raise ValueError(f"You must pass an image or a fig and ax")

    # Highlight boundaries if boundary_color is specified
    if segments is not None:

        # Convert named colors to RGB tuples if necessary
        boundary_color = (255, 0, 0)

        # Create a copy of the image to overlay the boundaries
        overlay = np.copy(image)

        # Identify the boundaries of the superpixels
        boundaries = find_boundaries(segments, mode='outer')

        # Put the colors in the overlay
        overlay[boundaries] = boundary_color

        # Plot the image with boundaries
        ax.imshow(overlay)

    elif image is not None:
        
        # Plot the image directly
        ax.imshow(image)

        # Add title and remove axes for cleaner visualization
        if title is not None:
            ax.set_title(title)
        ax.axis("off")

    # Save the plot to the desired file format
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:

        # Save as PNG temporarily
        fig.savefig(tmp_file.name, format="png")
        plt.close(fig)

        # Open the temporary PNG with Pillow
        temp_image = Image.open(tmp_file.name)

        # Convert and save as a standardized TIFF with sRGB ICC profile if selected
        if output_format.upper() == "TIFF":
            temp_image = ImageCms.profileToProfile(temp_image, srgb_profile, srgb_profile, outputMode="RGB")

        # Save the image in the specified format
        temp_image.save(output_file_name, format=output_format.upper())

##################################
### Core Functions
##################################

# Calculate Dynamic Threshold
def calculate_dynamic_threshold(image):
    """
    Calculate a dynamic threshold based on the standard deviation of pixel colors in the image.
    """

    # Flatten image to (num_pixels, 3)
    pixels = image.reshape(-1, 3)
    # Calculate standard deviation of each color channel
    std_dev = np.std(pixels, axis=0)

    # Use average std_dev to define threshold sensitivity
    dynamic_threshold = int(np.mean(std_dev))

    return dynamic_threshold

# Replace Black Pixels By Closest Non-black Neighbor
def replace_black_pixels(image):
    """
    Replace black pixels (BGR = [0, 0, 0]) with the nearest non-black pixel in the image.
    """

    # Find coordinates of non-black pixels
    height, width, channels = image.shape
    non_black_coords = np.array(
        [(x, y) for x in range(height) for y in range(width) if not np.all(image[x, y] == [0, 0, 0])]
    )
    non_black_pixels = np.array([image[x, y] for x, y in non_black_coords])

    # Find coordinates of black pixels
    black_coords = np.array(
        [(x, y) for x in range(height) for y in range(width) if np.all(image[x, y] == [0, 0, 0])]
    )

    # Image contains only black pixels
    if len(non_black_coords) == 0:
        raise ValueError("Image contains only black pixels.")

    # Create KDTree for fast nearest-neighbor search
    tree = KDTree(non_black_coords)
    for x, y in black_coords:
        nearest_idx = tree.query([x, y])[1]
        image[x, y] = non_black_pixels[nearest_idx]

    return image

# Perform Superpixel Segmentation using SLIC
def perform_superpixel_segmentation(lab_image, n_segments=200):
    """
    Perform superpixel segmentation on the LAB image using the SLIC algorithm.
    """

    # Apply SLIC for superpixel segmentation
    segments = slic(lab_image, n_segments=n_segments, compactness=30, start_label=0)
  
    return segments

def save_superpixel_visualization(image, segments, output_file_name, boundary_color="red", title="SLIC Segmentation"):
    """
    Save a visualization of SLIC superpixel segmentation with boundaries highlighted.

    Parameters:
        image (numpy.ndarray): The original image.
        segments (numpy.ndarray): The segmentation result from SLIC.
        output_file_name (str): Path to save the output image.
        boundary_color (str, optional): Color for the boundaries. Default is "red".
        title (str, optional): Title for the plot. Default is "SLIC Segmentation".
    """

    # Save the visualization using the save_plot function
    save_plot(image, output_file_name, segments=segments, title=title, color_space="BGR")

# Perform K-Means Clustering
def perform_kmeans_clustering(lab_image, segments=None, num_clusters=10, use_superpixels=False):
    """
    Perform K-Means clustering on the LAB image, initializing centroids using superpixel segmentation.

    Parameters:
        lab_image (numpy.ndarray): LAB image for clustering.
        segments (numpy.ndarray, optional): Superpixel segmentation labels.
        num_clusters (int): Number of K-means clusters.
        use_superpixels (bool): Whether to initialize centroids using superpixels.

    Returns:
        numpy.ndarray: Quantized image with K-means clustering.
        numpy.ndarray: Cluster centroids (LAB values).
        numpy.ndarray: Pixel counts per cluster.
    """
    pixels = lab_image.reshape(-1, 3)  # Flatten LAB image into a 2D array of pixels

    if use_superpixels:
        if segments is None:
            raise ValueError("Superpixel segments must be provided when use_superpixels is True.")

        # Compute mean LAB color for each superpixel segment
        unique_segments = np.unique(segments)
        initial_centroids = []
        for seg_val in unique_segments:
            mask = segments == seg_val
            mean_color = lab_image[mask].mean(axis=0)
            initial_centroids.append(mean_color)

        initial_centroids = np.array(initial_centroids)  # Convert to NumPy array

        # Limit centroids to the specified number of clusters
        if len(initial_centroids) > num_clusters:
            initial_centroids = initial_centroids[:num_clusters]

        # Perform K-means clustering with pre-defined centroids
        kmeans = KMeans(n_clusters=len(initial_centroids), init=initial_centroids, n_init=1, random_state=SEED)
        labels = kmeans.fit_predict(pixels)

        # Merge close centroids (optional)
        # Define a threshold for merging centroids
        merge_threshold = 10.0  # LAB distance threshold
        merged_centroids = []
        for i, center in enumerate(kmeans.cluster_centers_):
            if not any(np.linalg.norm(center - np.array(merged)) < merge_threshold for merged in merged_centroids):
                merged_centroids.append(center)

        # Create the quantized image
        quantized_pixels = np.round(kmeans.cluster_centers_).astype("uint8")[labels]
        quantized_image = quantized_pixels.reshape(lab_image.shape)

        # Centroids and pixel counts
        centroids = np.round(kmeans.cluster_centers_).astype("uint8")
        pixel_counts = np.bincount(labels, minlength=len(centroids))

    else:
        # Global K-means without superpixel initialization
        kmeans = KMeans(n_clusters=num_clusters, random_state=SEED)
        labels = kmeans.fit_predict(pixels)

        # Create the quantized image
        quantized_pixels = np.round(kmeans.cluster_centers_).astype("uint8")[labels]
        quantized_image = quantized_pixels.reshape(lab_image.shape)

        # Centroids and pixel counts
        centroids = np.round(kmeans.cluster_centers_).astype("uint8")
        pixel_counts = np.bincount(labels, minlength=num_clusters)

    return quantized_image, centroids, pixel_counts

# Save Quantized Image
def save_quantized_image(quantized_image, output_file_name, color_space="LAB", output_format="TIFF"):
    """
    Save a quantized image for visualization.

    Parameters:
        quantized_image (numpy.ndarray): The quantized image to save.
        output_file_name (str): The path to save the image.
        color_space (str, optional): The color space of the image (default: "LAB").
        output_format (str, optional): The format to save the image (default: "TIFF").
    """
    save_picture(output_file_name, quantized_image, n_rows=1, n_cols=1, color_space=color_space, output_format=output_format)

# Plot Clusters in Quantized Image
def plot_clusters_in_quantized_image(image, quantized_image, output_file_name, color_space="LAB", output_format="TIFF"):
    """
    Plot clusters in a quantized image, showing the original image pixels for each cluster.

    Parameters:
        image (numpy.ndarray): The original image.
        quantized_image (numpy.ndarray): The quantized image with cluster labels.
        output_file_name (str): Path to save the resulting image.
        color_space (str, optional): Color space of the input images ("LAB", "RGB"). Default is "LAB".
        output_format (str, optional): Format of the output file ("TIFF", "PNG", "JPG", etc.). Default is "TIFF".
    """
    # Convert the quantized image to RGB for visualization
    if color_space.upper() == "LAB":
        visual_image = cv2.cvtColor(quantized_image.astype("uint8"), cv2.COLOR_LAB2RGB)
    elif color_space.upper() == "RGB":
        visual_image = quantized_image
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

    # Identify unique clusters in LAB space
    unique_clusters = np.unique(quantized_image.reshape(-1, 3), axis=0)
    num_clusters = len(unique_clusters)

    # Calculate grid size
    n_rows = int(math.ceil(math.sqrt(num_clusters)))
    n_cols = int(math.ceil(num_clusters / n_rows))

    # Create masks for each cluster and generate masked images
    cluster_images = []
    for cluster_color in unique_clusters:
        # Create a mask for the current cluster (comparison in LAB space)
        mask = (quantized_image == cluster_color).all(axis=-1)

        # Create a blank image with only the cluster pixels (in RGB space)
        cluster_image = np.zeros_like(visual_image)
        cluster_image[mask] = visual_image[mask]

        cluster_images.append(cluster_image)

    # Save the grid of images
    save_picture(output_file_name, *cluster_images, n_rows=n_rows, n_cols=n_cols, color_space="RGB", output_format=output_format)

# Plot Cluster Colors
def plot_cluster_colors(cluster_centers, pixel_counts, hexadecimal_to_name, output_file):
    """
    Plot a grid displaying the color of each cluster from K-means with LAB and RGB values, 
    along with the total number of pixels in each cluster.

    Parameters:
        cluster_centers (numpy.ndarray): Cluster center colors in LAB format.
        pixel_counts (numpy.ndarray): Number of pixels per cluster.
        hexadecimal_to_name (dict): Dictionary to map hexadecimal color to color names.
        output_file (str): Path to save the resulting plot.
    """

    # Convert LAB cluster centers to RGB
    rgb_centers = cv2.cvtColor(cluster_centers[np.newaxis, :, :].astype("uint8"), cv2.COLOR_LAB2RGB)[0]

    # Calculate grid dimensions
    num_colors = len(cluster_centers)
    grid_size = ceil(num_colors**0.5)
    n_rows, n_cols = grid_size, grid_size

    # Create a blank canvas for display
    box_size = 100
    canvas = np.ones((n_rows * box_size, n_cols * box_size, 3), dtype=np.uint8) * 255

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(n_cols * 2, n_rows * 2))
    ax.imshow(canvas)
    ax.axis("off")

    # Iterate over clusters and fill grid
    for idx, (lab_color, rgb_color) in enumerate(zip(cluster_centers, rgb_centers)):
        l, a, b = lab_color
        r, g, b_rgb = rgb_color
        h, s, v = convert_rgb_to_hsv((r, g, b_rgb))
        c, m, y, k = rgb_to_cmyk((r, g, b_rgb))

        color_name = closest_color_name((r, g, b_rgb), hexadecimal_to_name)
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
            f"LAB: ({l:.1f}, {a:.1f}, {b:.1f})\n"
            f"CMYK: ({c:.1f}, {m:.1f}, {y:.1f}, {k:.1f})\n"
            f"HSV: ({h:.1f}, {s:.1f}, {v:.1f})"
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

    # Use `save_plot` to save the resulting visualization
    save_plot(canvas, fig=fig, ax=ax, output_file_name=output_file, title="Cluster Colors with Annotations", color_space="RGB")

# Calculate Color Distributions
def calculate_color_distributions(rgb_image, hsv_image, lab_image, quantized_image, cluster_centers, pixel_counts, hexadecimal_to_name, output_file):
    """
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
    """
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

# Plot Annotated Clusters in Quantized Image
def plot_annotated_clusters_in_quantized_image(image, quantized_image, centroids, hexadecimal_to_name, output_file_name, color_space="BGR", quantized_color_space="LAB", output_format="TIFF"):
    """
    Plot clusters in a quantized image with annotations showing the color name of each cluster.

    Parameters:
        image (numpy.ndarray): The original image.
        quantized_image (numpy.ndarray): The quantized image with cluster labels.
        centroids (numpy.ndarray): A list of centroid RGB values (one for each cluster).
        hexadecimal_to_name (dict): A dictionary mapping hex color codes to color names.
        output_file_name (str): Path to save the resulting image.
        color_space (str, optional): Color space of the input images ("BGR", "RGB"). Default is "BGR".
        output_format (str, optional): Format of the output file ("TIFF", "PNG", "JPG", etc.). Default is "TIFF".
    """

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
    n_rows = int(math.ceil(math.sqrt(num_clusters)))
    n_cols = int(math.ceil(num_clusters / n_rows))

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
    save_plot(None, output_file_name, fig=fig, ax=axs, color_space="RGB", output_format=output_format)

#########
# MAIN
#########

def main(srgb_profile, input_colors_file_name, input_folder, output_folder, output_file_name):
    """
    Main function.

    Parameters:
        srgb_profile (PIL.ImageCms.core.CmsProfile): RGB ICC Profile for TIFF Files.
        input_colors_file_name (string): File containing community color names.
        input_folder (string): Folder containing TIFF images.
        output_folder (string): Path to store plots and files.
        output_file_name (string): File to write color distributions.
    """

    time1 = perf_counter()

    # Create color name dictionary
    hexadecimal_to_name = load_hexadecimal_to_name(input_colors_file_name)

    time2 = perf_counter()
    print(f"1,2: {time2-time1:.6f}")

    # Iterate through kernel images
    output_file = open(output_file_name, "w")
    for file_name in os.listdir(input_folder):

        # Join file path
        file_path = os.path.join(input_folder, file_name)
        output_file.write(f"# {file_name}\n")
        print(file_name)

        # If not an image, continue
        if not file_name.lower().endswith(".tiff"):
            continue

        # Load the image
        image = cv2.imread(file_path)
        image = replace_black_pixels(image)  # Step 1: Replace black (annotation) pixels by closest neighbors
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Generate RGB image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Generate HSV image
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Generate LAB image

        time12 = 9
        time2 = min(time2, time12)
        time3 = perf_counter()
        print(f"2,3: {time3-time2:.6f}")

        # Step 2: Calculate dynamic threshold
        threshold = calculate_dynamic_threshold(image)

        time4 = perf_counter()
        print(f"3,4: {time4-time3:.6f}")

        # Step 3: Perform Superpixel Segmentation
        superpixel_segments = perform_superpixel_segmentation(lab_image, n_segments=threshold * 10)

        time5 = perf_counter()
        print(f"4,5: {time5-time4:.6f}")

        # Step 4: Save Superpixel Visualization
        superpixel_plot_file_name = os.path.join(output_folder, f"{file_name}_superpixels.tiff")
        save_superpixel_visualization(image, superpixel_segments, superpixel_plot_file_name)

        time6 = perf_counter()
        print(f"5,6: {time6-time5:.6f}")

        # Step 5: Perform K-Means Clustering
        quantized_image, centroids, pixel_counts = perform_kmeans_clustering(lab_image, segments=superpixel_segments, use_superpixels=True)

        time7 = perf_counter()
        print(f"6,7: {time7-time6:.6f}")

        #print("Quantized Image Max Cluster Index:", np.max(quantized_image))
        #print("Number of Centroids:", len(centroids))
        #print("Total Pixels:", np.sum(pixel_counts))
        #print("Image Pixels:", image.shape[0] * image.shape[1])

        # Step 6: Save Quantized Image
        quantized_plot_file_name = os.path.join(output_folder, f"{file_name}_quantized.tiff")
        save_quantized_image(quantized_image, quantized_plot_file_name)

        time8 = perf_counter()
        print(f"7,8: {time8-time7:.6f}")

        # Step 7: Plot cluster colors in the real image
        output_clusters_in_quantized_image = os.path.join(output_folder, f"{file_name}_clusters_in_quantized.tiff")
        plot_clusters_in_quantized_image(lab_image, quantized_image, output_clusters_in_quantized_image, color_space="LAB")

        time9 = perf_counter()
        print(f"8,9: {time9-time8:.6f}")

        # Step 8: Save the plot of cluster colors
        plot_cluster_colors_file = os.path.join(output_folder, f"{file_name}_cluster_colors.tiff")
        plot_cluster_colors(centroids, pixel_counts, hexadecimal_to_name, plot_cluster_colors_file)

        time10 = perf_counter()
        print(f"9,10: {time10-time9:.6f}")

        # Step 9: Write RGB/HSV/LAB distributions to the file
        distributions = calculate_color_distributions(rgb_image, hsv_image, lab_image, quantized_image, centroids,
                                                      pixel_counts, hexadecimal_to_name, output_file)

        time11 = perf_counter()
        print(f"10,11: {time11-time10:.6f}")

        # Step 10: Plot cluster colors in the real image with Centroid Color Name annotation
        output_annotated_color_name_image = os.path.join(output_folder, f"{file_name}_annotated_color_name.tiff")
        plot_annotated_clusters_in_quantized_image(image, quantized_image, centroids, hexadecimal_to_name, output_annotated_color_name_image)

        time12 = perf_counter()
        print(f"11,12: {time12-time11:.6f}")

    # Closing file    
    output_file.close()

if __name__ == "__main__":

    # Input Output Files
    input_colors_file_name = "/Users/egg/Projects/Stainalyzer/data/colornames.txt"
    input_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/tiff/"
    output_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining8/"
    output_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining8/results.txt"
    os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

    # Create the sRGB ICC profile
    srgb_profile = ImageCms.createProfile("sRGB")

    # Calling main
    main(srgb_profile, input_colors_file_name, input_folder, output_folder, output_file_name)
