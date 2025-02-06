
# Try to use Superpixel Algorithm (SLIC) followed by K-Means Clustering for global color quantization.
# Additionally, it includes a dynamic threshold mechanism based on image color variability.

import io
import os
import cv2
import tempfile
import webcolors
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
from PIL import Image, ImageCms

# Input Output Files
input_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/tiff/"
output_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining6/"
output_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining6/results.txt"
output_cluster_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining6/color_cluster.txt"
os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

# Create the sRGB ICC profile
srgb_profile = ImageCms.createProfile("sRGB")

# Dynamic Threshold Function (Based on Color Variability)
def calculate_dynamic_threshold(image):

    # Flatten image to (num_pixels, 3)
    pixels = image.reshape(-1, 3)
    # Calculate standard deviation of each color channel
    std_dev = np.std(pixels, axis=0)

    # Use average std_dev to define threshold sensitivity
    dynamic_threshold = int(np.mean(std_dev))

    return dynamic_threshold

# Function to replace black pixels (which are only the size annotation of the image) by its closest neighbor pixel
def replace_black_pixels(image):
    """Replace black pixels (BGR = 0, 0, 0) with the nearest non-black pixel in a BGR image."""
    height, width, channels = image.shape
    # Find coordinates of non-black pixels
    non_black_coords = np.array(
        [(x, y) for x in range(height) for y in range(width) if not np.all(image[x, y] == [0, 0, 0])]
    )
    non_black_pixels = np.array([image[x, y] for x, y in non_black_coords])
    # Find coordinates of black pixels
    black_coords = np.array(
        [(x, y) for x in range(height) for y in range(width) if np.all(image[x, y] == [0, 0, 0])]
    )

    if len(non_black_coords) == 0:
        raise ValueError("Image contains only black pixels.")

    # Create KDTree for fast nearest-neighbor search
    tree = KDTree(non_black_coords)
    for x, y in black_coords:
        nearest_idx = tree.query([x, y])[1]
        image[x, y] = non_black_pixels[nearest_idx]

    return image

# Superpixel Segmentation Function (SLIC)
def perform_superpixel_segmentation(lab_image, n_segments=200):

    # Apply SLIC for superpixel segmentation
    segments = slic(lab_image, n_segments=n_segments, compactness=10, start_label=0)
  
    return segments

# K-Means Clustering Function
def perform_kmeans_clustering(image, num_clusters=10):

    # Flatten image to (num_pixels, 3)
    pixels = image.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state = 42)

    labels = kmeans.fit_predict(pixels)
    # Replace pixel colors with cluster centers
    quantized_image = kmeans.cluster_centers_.astype("uint8")[labels].reshape(image.shape)
    return quantized_image, kmeans.cluster_centers_

# Save image as standardized TIFF
def save_as_tiff(image, file_path):
    """
    Save an image as a TIFF with a standard sRGB ICC profile.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image = ImageCms.profileToProfile(pil_image, srgb_profile, srgb_profile, outputMode="RGB")
    pil_image.save(file_path, format="TIFF")

# Save Superpixel Visualization
def save_superpixel_visualization(image, segments, file_path):

    # Create a copy of the image to overlay boundaries
    overlay = image.copy()

    # Draw boundaries for superpixels
    for i in np.unique(segments):
        mask = segments == i
        overlay[mask] = overlay[mask].mean(axis=0)  # Assign mean color to region

    save_as_tiff(overlay, file_path)

# Save Quantized Image
def save_quantized_image(image, file_path):
    save_as_tiff(image, file_path)

# Plot Cluster Color's Centroids
def plot_cluster_colors(quantized_image, cluster_centers, output_file):
    """
    Plot a single figure showing all unique cluster colors with their RGB, CMYK, and HSV values.
    
    Parameters:
    - quantized_image: The quantized image output by the clustering algorithm.
    - cluster_centers: The cluster centers in RGB format.
    - output_file: Path to save the output plot.
    """

    # Get unique colors from the cluster centers
    unique_colors = np.unique(cluster_centers, axis=0).astype(int)
    num_colors = len(unique_colors)
    
    # Determine grid size for plot
    grid_size = ceil(num_colors**0.5)
    box_size = 100  # Size of each color box
    
    # Create canvas
    canvas_width = grid_size * box_size
    canvas_height = grid_size * box_size
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White background
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for idx, color in enumerate(unique_colors):

        # Calculate position
        row = idx // grid_size
        col = idx % grid_size
        x_start = col * box_size
        y_start = row * box_size
        
        # Fill the square with the cluster color
        canvas[y_start:y_start+box_size, x_start:x_start+box_size] = color
        
        # Calculate HSV and CMYK values
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]

        r, g, b = map(int, color)  # Convert RGB values to plain integers
        h, s, v = map(int, hsv_color)  # Convert HSV values to plain integers
        c, m, y, k = rgb_to_cmyk(r, g, b)

        # Add annotations
        text = (
            f"RGB: ({r}, {g}, {b})\n"
            f"CMYK: ({c:.2f}, {m:.2f}, {y:.2f}, {k:.2f})\n"
            f"HSV: ({h}, {s}, {v})"
        )
        ax.text(
            x_start + box_size // 2,
            y_start + box_size // 2,
            text,
            fontsize=6,
            ha="center", 
            va="center",
            color="black",
            backgroundcolor="white"
        )
    
    # Display the image
    ax.imshow(canvas)
    ax.axis("off")
    plt.title("Cluster Colors with Annotations")
    plt.tight_layout()
    # Step 1: Save as a temporary image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        plt.savefig(tmp_file.name, format="png")
        plt.close()

        # Step 2: Open the temporary file with Pillow
        temp_image = Image.open(tmp_file.name)

        # Step 3: Convert and save as a standardized TIFF with sRGB ICC profile
        temp_image = ImageCms.profileToProfile(temp_image, srgb_profile, srgb_profile, outputMode="RGB")
        temp_image.save(output_file, format="TIFF")

def rgb_to_cmyk(r, g, b):
    """Convert RGB to CMYK."""
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

def format_value(e):
    return str(round(e, 4)) if isinstance(e, float) else str(e)

def write_hsv_distributions(hsv_image, quantized_image, cluster_centers, file_name, output_file, output_cluster_file):
    """
    Write HSV distributions (mean and std) for each cluster to the output file.
    """

    output_file.write(f"{file_name}\n")  # File name header

    for cluster_idx, cluster_color in enumerate(cluster_centers):

        # Get all pixels assigned to this cluster
        cluster_color = cluster_color.astype(quantized_image.dtype) # Make sure they have the same dtype
        mask = (quantized_image == cluster_color).all(axis=-1)
        cluster_pixels = hsv_image[mask]  # Get HSV pixels for the current cluster

        if cluster_pixels.size == 0:
            continue  # Skip empty clusters
        
        # Calculate mean and std for H, S, V
        hue_mean, hue_std = cluster_pixels[:, 0].mean(), cluster_pixels[:, 0].std()
        sat_mean, sat_std = cluster_pixels[:, 1].mean(), cluster_pixels[:, 1].std()
        val_mean, val_std = cluster_pixels[:, 2].mean(), cluster_pixels[:, 2].std()

        # Convert cluster color (RGB) to a name
        cluster_color_name = closest_color_name(cluster_color)

        # Write the distribution to the output file
        output_file.write(f"Cluster: {cluster_idx}\n")
        output_file.write(f"Color Name: {cluster_color_name}\n")
        output_file.write(
            f"{cluster_color_name.upper().replace(' ', '_')}_PARAMS = {{\n"
            f'    "hue": ({hue_mean}, {hue_std}),\n'
            f'    "saturation": ({sat_mean}, {sat_std}),\n'
            f'    "value": ({val_mean}, {val_std})\n'
            "}\n\n"
        )
        output_file.write("\n")  # Add extra newline for separation

        # Write formatted distribution
        output_cluster_file.write("\t".join(format_value(e) for e in [cluster_color_name, hue_mean, hue_std, sat_mean, 
                                                                      sat_std, val_mean, val_std]) + "\n")
def closest_color_name(rgb_color):

    try:
        # Attempt to find an exact color name
        return webcolors.rgb_to_name(rgb_color)
    except ValueError:
        # If there's no exact match, find the closest color
        closest_name = None
        min_distance = float('inf')
        for name in webcolors.names("css3"):
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            rd = (r_c - rgb_color[0]) ** 2
            gd = (g_c - rgb_color[1]) ** 2
            bd = (b_c - rgb_color[2]) ** 2
            distance = rd + gd + bd
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        return closest_name

#########
# MAIN
#########

# Iterate through kernel images
output_file = open(output_file_name, "w")
output_cluster_file = open(output_cluster_file_name, "w")
for file_name in os.listdir(input_folder):

    # Join file path
    file_path = os.path.join(input_folder, file_name)
    output_cluster_file.write(f"# {file_name}\n")
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

    # Step 2: Calculate dynamic threshold
    threshold = calculate_dynamic_threshold(image)

    # Step 3: Perform Superpixel Segmentation
    superpixel_segments = perform_superpixel_segmentation(lab_image, n_segments = threshold * 10)

    # Step 4: Save Superpixel Visualization
    superpixel_path = os.path.join(output_folder, f"{file_name}_superpixels.tiff")
    save_superpixel_visualization(image, superpixel_segments, superpixel_path)

    # Step 5: Perform K-Means Clustering
    quantized_image, cluster_centers = perform_kmeans_clustering(image, num_clusters=threshold)
    
    # Step 6: Save Quantized Image
    quantized_path = os.path.join(output_folder, f"{file_name}_quantized.tiff")
    save_quantized_image(quantized_image, quantized_path)

    # Step 7: Write Results to Output File
    output_file.write(f"Processed {file_name}:\n")
    output_file.write(f"  - Superpixels saved to: {superpixel_path}\n")
    output_file.write(f"  - Quantized image saved to: {quantized_path}\n")
    output_file.write(f"  - Cluster Centers: {cluster_centers.tolist()}\n\n")

    # Step 8: Save the plot of cluster colors
    plot_file = os.path.join(output_folder, f"{file_name}_cluster_colors.tiff")
    plot_cluster_colors(quantized_image, cluster_centers, plot_file)

    # Step 9: Write HSV distributions to the file
    write_hsv_distributions(hsv_image, quantized_image, cluster_centers, file_name, output_file, output_cluster_file)

# Closing file    
output_file.close()
output_cluster_file.close()



