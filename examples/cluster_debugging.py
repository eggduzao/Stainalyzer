
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
import struct
import tempfile
import numpy as np
import seaborn as sns
from math import ceil
from PIL import Image, ImageCms
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
from skimage.segmentation import mark_boundaries

##################################
### File Names
##################################

# Input Output Files
input_colors_file_name = "/Users/egg/Projects/Stainalyzer/data/colornames.txt"
input_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/tiff/"
output_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/cluster_debugging/"
os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

# Create the sRGB ICC profile
srgb_profile = ImageCms.createProfile("sRGB")

##################################
### Centralized Color Utilities
##################################

# Save as TIFF
def save_as_tiff(image, file_path):
    """
    General function to save an image as a TIFF with a standard sRGB ICC profile.
    """

    # Save as TIFF using PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image = ImageCms.profileToProfile(pil_image, srgb_profile, srgb_profile, outputMode="RGB")
    pil_image.save(file_path, format="TIFF")

def convert_opencv_to_standard(color_array, color_space="HSV"):
    """
    Convert an OpenCV color array to its standard equivalent.

    Parameters:
        color_array (numpy.ndarray): The color array in OpenCV format (e.g., HSV or LAB).
        color_space (str): The color space of the input array ("HSV" or "LAB").

    Returns:
        numpy.ndarray: The converted color array in standard format.
    """
    color_array = np.array(color_array, dtype=np.float64)
    if color_space.upper() == "HSV":
        # Convert OpenCV HSV (H: 0-179, S: 0-255, V: 0-255) to standard HSV (H: 0-360, S: 0-1, V: 0-1)
        h, s, v = color_array
        h_standard = h * 2  # Scale H from [0, 179] to [0, 360]
        s_standard = s / 255.0  # Scale S from [0, 255] to [0, 100]
        v_standard = v / 255.0  # Scale V from [0, 255] to [0, 100]
        return np.array([h_standard, s_standard, v_standard])

    elif color_space.upper() == "LAB":
        # Convert OpenCV LAB (L: 0-255, A: 0-255, B: 0-255) to standard LAB (L: 0-100, A: -128 to 127, B: -128 to 127)
        l, a, b = color_array
        l_standard = l * 100 / 255.0  # Scale L from [0, 255] to [0, 100]
        a_standard = a - 128  # Shift A from [0, 255] to [-128, 127]
        b_standard = b - 128  # Shift B from [0, 255] to [-128, 127]
        return np.array([l_standard, a_standard, b_standard])

def display_image(image, output_file_name, color_space="BGR", title="Image"):
    """
    Display an image in any color space using matplotlib.

    Parameters:
        image (numpy.ndarray): The image to be displayed.
        color_space (str): The color space of the image (e.g., "BGR", "RGB", "HSV", "LAB").
        title (str): Title of the plot.
    """
    if color_space == "BGR":
        # Convert BGR to RGB for correct display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == "HSV":
        # Convert HSV to RGB for correct display
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    elif color_space == "LAB":
        # Convert LAB to RGB for correct display
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.savefig(output_file_name, format="png")
    plt.close()

def plot_slic_segmentation(image, output_file_name, segments, title="SLIC Segmentation"):
    """
    Plot the SLIC segmentation boundaries on the original image.
    
    Parameters:
        image (numpy.ndarray): The original image.
        segments (numpy.ndarray): The segmentation result from SLIC.
        title (str): The title of the plot.
    """
    # Convert image to RGB if it's in another color space (e.g., BGR)
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    # Mark the boundaries on the original image
    marked_image = mark_boundaries(image_rgb, segments, color=(1, 0, 0))

    # Plot the result
    plt.figure(figsize=(10, 10))
    plt.imshow(marked_image)
    plt.title(title)
    plt.axis("off")
    plt.savefig(output_file_name, format="png")
    plt.close()

import cv2
import matplotlib.pyplot as plt

def compare_and_save_images(image1, image2, output_image_file_name):
    """
    Compare two images by displaying them side by side and save the plot to a file.

    Parameters:
        image1 (numpy.ndarray): The first image (can be any type, e.g., BGR, RGB, etc.).
        image2 (numpy.ndarray): The second image (can be any type, e.g., BGR, RGB, etc.).
        output_image_file_name (str): The file name to save the comparison plot.
    """
    # Convert images to RGB if necessary
    if len(image1.shape) == 3 and image1.shape[2] == 3:  # Check if the image has color channels
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    if len(image2.shape) == 3 and image2.shape[2] == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    
    # Plot the first image
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title("Image 1")
    plt.axis("off")
    
    # Plot the second image
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title("Image 2")
    plt.axis("off")
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_image_file_name, format='png')
    plt.close()

##################################
### Core Functions
##################################

# Calculate Dynamic Threshold
def calculate_dynamic_threshold(image, output_file):
    """
    Calculate a dynamic threshold based on the standard deviation of pixel colors in the image.
    """

    # Flatten image to (num_pixels, 3)
    pixels = image.reshape(-1, 3)

    output_file.write(str(pixels))
    output_file.write("\n##################################################################################\n\n")

    # Calculate standard deviation of each color channel
    std_dev = np.std(pixels, axis=0)

    output_file.write(str(std_dev))
    output_file.write("\n##################################################################################\n\n")

    # Use average std_dev to define threshold sensitivity
    dynamic_threshold = int(np.mean(std_dev))
    output_file.write(str(dynamic_threshold))
    output_file.write("\n##################################################################################\n\n")

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
def perform_superpixel_segmentation(lab_image, output_file, n_segments=200):
    """
    Perform superpixel segmentation on the LAB image using the SLIC algorithm.
    """

    # Apply SLIC for superpixel segmentation
    segments = slic(lab_image, n_segments=n_segments, compactness=30, start_label=0)
  
    output_file.write(str(segments))
    output_file.write("\n##################################################################################\n\n")

    return segments

# K-Means Clustering of Colors
def perform_kmeans_clustering(image, output_file, num_clusters=10):
    """
    Perform K-Means clustering on the image to identify color clusters
    and update the centroids and pixel counts based on the quantized image.
    """
    # Flatten image to (num_pixels, 3)
    pixels = image.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)

    # Initial pixel counts
    pixel_counts = np.bincount(labels, minlength=num_clusters)
    output_file.write(str(pixel_counts))
    output_file.write("\n##################################################################################\n\n")

    # Replace pixel colors with cluster centers
    quantized_image = kmeans.cluster_centers_.astype("uint8")[labels].reshape(image.shape)
    output_image_file_name = os.path.join(output_folder, "quantized_image.png")
    display_image(quantized_image, output_image_file_name, color_space="BGR", title="Quantized Image")

    # Update centroids and pixel counts based on quantized image
    unique_colors, updated_pixel_counts = np.unique(quantized_image.reshape(-1, 3), axis=0, return_counts=True)
    output_file.write(str(unique_colors))
    output_file.write("\n##################################################################################\n\n")
    output_file.write(str(updated_pixel_counts))
    output_file.write("\n##################################################################################\n\n")

    updated_centroids = unique_colors.astype("uint8")
    output_file.write(str(updated_centroids))
    output_file.write("\n##################################################################################\n\n")

    # Return updated elements
    return quantized_image, updated_centroids, updated_pixel_counts

#########
# MAIN
#########

# Output file for printing
output_file_name = os.path.join(output_folder, "results.txt")
output_file = open(output_file_name, "w")

# Iterate through kernel images
for file_name in os.listdir(input_folder):

    # Join file path
    file_path = os.path.join(input_folder, file_name)
    output_file.write(f"# {file_name}\n")
    output_file.write("\n##################################################################################\n\n")

    # If not an image, continue
    if not file_name.lower().endswith(".tiff"):
        continue

    # Load the image
    image = cv2.imread(file_path)
    output_file.write(str(image[0][0]))
    output_file.write("\n##################################################################################\n\n")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Generate RGB image
    output_file.write(str(rgb_image[0][0]))
    output_file.write("\n##################################################################################\n\n")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Generate HSV image
    output_file.write(str(hsv_image[0][0]))
    output_file.write(str(convert_opencv_to_standard(hsv_image[0][0], "HSV")))
    output_file.write("\n##################################################################################\n\n")
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Generate LAB image
    output_file.write(str(lab_image[0][0]))
    output_file.write(str(convert_opencv_to_standard(lab_image[0][0], "LAB")))
    output_file.write("\n##################################################################################\n\n")

    # If not an image, continue
    if not file_name.lower().endswith(".tiff"):
        continue

    # Load the image
    image = cv2.imread(file_path)
    image = replace_black_pixels(image)  # Step 1: Replace black (annotation) pixels by closest neighbors
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Generate RGB image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Generate HSV image
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Generate LAB image

    # Display the original BGR image
    #output_image_file_name = os.path.join(output_folder, "BGR.png")
    #display_image(image, output_image_file_name, color_space="BGR", title="Original BGR Image")

    # Display the RGB image
    #display_image(rgb_image, output_image_file_name, color_space="RGB", title="RGB Image")

    # Display the HSV image
    #display_image(hsv_image, output_image_file_name, color_space="HSV", title="HSV Image")

    # Display the LAB image
    #display_image(lab_image, output_image_file_name, color_space="LAB", title="LAB Image")

    # Step 2.1. Image Denoising
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    lab_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2LAB)
    output_image_file_name = os.path.join(output_folder, "Denoise.png")
    compare_and_save_images(image, denoised_image, output_image_file_name)

    # Step 2: Calculate dynamic threshold
    threshold = calculate_dynamic_threshold(image, output_file)

    # Step 3: Perform Superpixel Segmentation
    superpixel_segments = perform_superpixel_segmentation(lab_image, output_file, n_segments = threshold * 10)
    output_file.write(str(len(np.unique(superpixel_segments))))
    output_file.write("\n##################################################################################\n\n")
    output_image_file_name = os.path.join(output_folder, "SLIC_30_D.png")
    plot_slic_segmentation(image, output_image_file_name, superpixel_segments)

    # Step 5: Perform K-Means Clustering
    quantized_image, cluster_centers, pixel_counts = perform_kmeans_clustering(image, output_file, num_clusters=threshold)

#superpixel_means = [image[superpixel_segments == label].mean(axis=0) for label in np.unique(superpixel_segments)]
#kmeans.fit(superpixel_means)  # Cluster superpixel averages

# Closing file    
output_file.close()



