
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

# Import required libraries
import io
import os
import cv2
import struct
import tempfile
import webcolors
import numpy as np
import seaborn as sns
from math import ceil
from PIL import Image, ImageCms
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from scipy.spatial.distance import cdist

# Input Output Files
input_colors_file_name = "/Users/egg/Projects/Stainalyzer/data/colornames.txt"
input_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/tiff/"
output_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining7/"
output_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining7/results.txt"
output_cluster_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining7/color_cluster.txt"
os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

# Create the sRGB ICC profile
srgb_profile = ImageCms.createProfile("sRGB")

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
    segments = slic(lab_image, n_segments=n_segments, compactness=10, start_label=0)
  
    return segments

# K-Means Clustering of Colors
def perform_kmeans_clustering(image, num_clusters=10):
    """
    Perform K-Means clustering on the image to identify color clusters.
    """

    # Flatten image to (num_pixels, 3)
    pixels = image.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state = 42)
    labels = kmeans.fit_predict(pixels)

    # Getting pixel counts
    pixel_counts = np.bincount(labels, minlength=num_clusters)

    # Replace pixel colors with cluster centers
    quantized_image = kmeans.cluster_centers_.astype("uint8")[labels].reshape(image.shape)
    return quantized_image, kmeans.cluster_centers_, pixel_counts

# Save Superpixel Visualization
def save_superpixel_visualization(image, segments, file_path):
    """
    Save a visualization of the superpixel segmentation in a TIFF file.
    """

    # Save superpixel as TIFF
    overlay = image.copy()
    for i in np.unique(segments):
        mask = segments == i
        overlay[mask] = overlay[mask].mean(axis=0)
    save_as_tiff(overlay, file_path)

# Save as TIFF
def save_as_tiff(image, file_path):
    """
    General function to save an image as a TIFF with a standard sRGB ICC profile.
    """

    # Save as TIFF using PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image = ImageCms.profileToProfile(pil_image, srgb_profile, srgb_profile, outputMode="RGB")
    pil_image.save(file_path, format="TIFF")

# Save Quantized Image
def save_quantized_image(image, file_path):
    """
    Save the quantized image as a standardized TIFF.
    """

    # Save quantized image as TIFF
    save_as_tiff(image, file_path)

# Plot Color Clusters
def plot_cluster_colors(quantized_image, cluster_centers, hexadecimal_to_name, output_file, pixel_counts):
    """
    Plot a single figure showing all unique cluster colors with their RGB, CMYK, and HSV values,
    and the total number of pixels per cluster.
    """

    # Check sizes
    unique_colors = np.unique(cluster_centers, axis=0).astype(int)
    num_colors = len(unique_colors)
    grid_size = ceil(num_colors**0.5)
    box_size = 100
    canvas_width = grid_size * box_size
    canvas_height = grid_size * box_size
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Start image
    fig, ax = plt.subplots(figsize=(10, 10))

    # Iterate on clusters
    for idx, color in enumerate(unique_colors):

        # Grid locations and sizes
        row = idx // grid_size
        col = idx % grid_size
        x_start = col * box_size
        y_start = row * box_size

        # Put color into canvas
        canvas[y_start:y_start + box_size, x_start:x_start + box_size] = color

        # Calculate colors
        r, g, b = map(int, color)
        webcolor_name = closest_color_name(color, hexadecimal_to_name)
        total_pixels = pixel_counts[idx]
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]

        # Calculate HSV and CYMK
        h, s, v = map(int, hsv_color)
        c, m, y, k = rgb_to_cmyk(r, g, b)

        # Text to plot in the center
        text = (
            f"{webcolor_name}\nPixels: {total_pixels}"
            f"RGB: ({r:.1f}, {g:.1f}, {b:.1f})\n"
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

    """

        text = f"{webcolor_name}\nPixels: {total_pixels}"
        ax.text(
            x_start + box_size // 2,
            y_start + box_size // 2,
            text,
            fontsize=8,
            ha="center",
            va="center",
            color="black",
            backgroundcolor="white"
        )

    # Finish and save image
    ax.imshow(canvas)
    ax.axis("off")
    plt.title("Cluster Colors with Annotations")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    """

# Convert RGB to CMYK
def rgb_to_cmyk(r, g, b):
    """
    Convert RGB color values to CMYK.
    """
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

# Plot Violin Clusters
def plot_violin_clusters(hsv_image, quantized_image, cluster_centers, hexadecimal_to_name, output_file_prefix):
    """
    Create violin plots for Hue, Saturation, and Value distributions for each cluster.
    Includes annotations for average HSV, total pixel count, and min/max channel values.
    """

    # Start pixel data
    cluster_pixel_data = []

    # Collect data for each cluster
    for idx, cluster_color in enumerate(cluster_centers):
        cluster_color = cluster_color.astype(quantized_image.dtype)
        mask = (quantized_image == cluster_color).all(axis=-1)
        cluster_pixels = hsv_image[mask]
        cluster_pixel_data.append((idx, cluster_color, cluster_pixels, mask.sum()))

    # Create violin plots
    for channel_idx, channel_name in enumerate(['Hue', 'Saturation', 'Value']):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Iterate on clusters
        for idx, (cluster_idx, cluster_color, cluster_pixels, total_pixels) in enumerate(cluster_pixel_data):
            if cluster_pixels.size == 0:
                continue

            # Get values and colors
            channel_values = cluster_pixels[:, channel_idx]
            min_val, max_val = channel_values.min(), channel_values.max()
            avg_h, avg_s, avg_v = cluster_pixels[:, 0].mean(), cluster_pixels[:, 1].mean(), cluster_pixels[:, 2].mean()
            webcolor_name = closest_color_name(cluster_color, hexadecimal_to_name)

            # Add violin plot
            sns.violinplot(data=[channel_values], ax=ax, inner='point', color=cluster_color / 255, width=0.8)
            ax.text(idx, channel_values.max() + 10, f"{webcolor_name}\nAvg HSV: ({avg_h:.1f}, {avg_s:.1f}, {avg_v:.1f})\n"
                                                    f"Pixels: {total_pixels}",
                    ha='center', va='bottom', fontsize=8, color='black')

            # Add squares for min and max values
            ax.add_patch(plt.Rectangle((idx - 0.3, min_val - 5), 0.2, 10, color=cluster_color / 255))
            ax.add_patch(plt.Rectangle((idx + 0.1, max_val - 5), 0.2, 10, color=cluster_color / 255))

        # Finish and save image
        ax.set_title(f"{channel_name} Violin Plots per Cluster")
        ax.set_xlabel("Clusters")
        ax.set_ylabel(f"{channel_name} Values")
        plt.xticks(range(len(cluster_centers)), [f"Cluster {i}" for i in range(len(cluster_centers))])
        plt.tight_layout()
        plt.savefig(f"{output_file_prefix}_{channel_name.lower()}_violin.png")
        plt.close()

# Plot Masked Clusters
def plot_masked_clusters(image, quantized_image, cluster_centers, hexadecimal_to_name, output_file_prefix):
    """
    Create a plot of the original image masked by each cluster.
    Displays each masked image alongside its cluster name and total pixel count.
    """

    # Check sizes
    unique_colors = np.unique(cluster_centers, axis=0).astype(int)
    num_colors = len(unique_colors)
    grid_size = ceil(num_colors**0.5)
    box_size = image.shape[0] // grid_size
    canvas_width = grid_size * box_size
    canvas_height = grid_size * box_size

    # Start image
    fig, ax = plt.subplots(figsize=(15, 15))
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Iterate on clusters
    for idx, color in enumerate(unique_colors):
        row = idx // grid_size
        col = idx % grid_size
        x_start = col * box_size
        y_start = row * box_size

        # Mask the image for this cluster
        mask = (quantized_image == color).all(axis=-1)
        masked_image = np.zeros_like(image)
        masked_image[mask] = image[mask]

        # Resize the masked image to fit in the grid
        resized_mask = cv2.resize(masked_image, (box_size, box_size), interpolation=cv2.INTER_AREA)
        canvas[y_start:y_start + box_size, x_start:x_start + box_size] = resized_mask

        # Annotate the cluster color name and pixel count
        webcolor_name = closest_color_name(color, hexadecimal_to_name)
        total_pixels = mask.sum()
        ax.text(
            x_start + box_size // 2,
            y_start + box_size // 2 + 10,
            f"{webcolor_name}\nPixels: {total_pixels}",
            fontsize=8,
            ha="center",
            va="center",
            color="black",
            backgroundcolor="white"
        )

    # Finish and save image
    ax.imshow(canvas)
    ax.axis("off")
    plt.title("Masked Images by Cluster Color")
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_masked_clusters.png")
    plt.close()

# Write HSV Distributions
def write_hsv_distributions(hsv_image, quantized_image, cluster_centers, file_name, hexadecimal_to_name, output_file, output_cluster_file):
    """
    Write HSV distributions and pixel counts for each cluster to output files.
    """

    # File name header
    output_file.write(f"{file_name}\n")  

    # Iterate on the cluster centers
    for cluster_idx, cluster_color in enumerate(cluster_centers):

        # Get all pixels assigned to this cluster
        cluster_color = cluster_color.astype(quantized_image.dtype) # Make sure they have the same dtype
        mask = (quantized_image == cluster_color).all(axis=-1)

        # Get HSV pixels for the current cluster
        cluster_pixels = hsv_image[mask]

        # Disregard empty clusters
        if cluster_pixels.size == 0:
            continue

        # Calculate mean and std for H, S, V
        hue_mean, hue_std = cluster_pixels[:, 0].mean(), cluster_pixels[:, 0].std()
        sat_mean, sat_std = cluster_pixels[:, 1].mean(), cluster_pixels[:, 1].std()
        val_mean, val_std = cluster_pixels[:, 2].mean(), cluster_pixels[:, 2].std()

        # Sum total number of pixels for cluster color
        total_pixels = mask.sum()

        # Convert cluster color (RGB) to a name
        cluster_color_name = closest_color_name(cluster_color, hexadecimal_to_name)

        # Write to output file
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
        output_cluster_file.write("\t".join(map(str, [
            cluster_color_name, hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std, total_pixels
        ])) + "\n")

def load_hexadecimal_to_name(file_path):
    """
    Load the hexadecimal-to-name mapping from a file.
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
        # Convert hex_key to RGB
        r = float(int(hex_key[0:2], 16))
        g = float(int(hex_key[2:4], 16))
        b = float(int(hex_key[4:6], 16))
        # Calculate Manhattan distance
        distance = abs(r - rgb_color[0]) + abs(g - rgb_color[1]) + abs(b - rgb_color[2])
        if distance < min_distance:
            min_distance = distance
            closest_name = name

    return closest_name

#########
# MAIN
#########

# Create color name dictionary
hexadecimal_to_name = load_hexadecimal_to_name(input_colors_file_name)

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
    quantized_image, cluster_centers, pixel_counts = perform_kmeans_clustering(image, num_clusters=threshold)

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
    plot_cluster_colors(quantized_image, cluster_centers, hexadecimal_to_name, plot_file, pixel_counts)

    # Step 9: Write HSV distributions to the file
    write_hsv_distributions(hsv_image, quantized_image, cluster_centers, file_name, hexadecimal_to_name, output_file, output_cluster_file)

# Closing file    
output_file.close()
output_cluster_file.close()



