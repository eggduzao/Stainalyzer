
# Try to use Superpixel Algorithm (SLIC) followed by K-Means Clustering for global color quantization.
# Additionally, it includes a dynamic threshold mechanism based on image color variability.

# Import packages
import os
import cv2
import webcolors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from scipy.spatial.distance import cdist

# Input Output Files
input_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/"
output_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining4/"
output_file_name = "/Users/egg/Projects/Stainalyzer/examples/output/results.txt"
os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

# Dynamic Threshold Function (Based on Color Variability)
def calculate_dynamic_threshold(image):

    # Flatten image to (num_pixels, 3)
    pixels = image.reshape(-1, 3)
    # Calculate standard deviation of each color channel
    std_dev = np.std(pixels, axis=0)
    print(std_dev)

    # Use average std_dev to define threshold sensitivity
    #dynamic_threshold = max(5, int(np.mean(std_dev) / 10))  # Scale factor
    #dynamic_threshold = max(3, int(np.mean(std_dev) / 15)) # Reduce Colors (more conserved quantization) = This will result in fewer clusters for K-Means.
    #dynamic_threshold = max(10, int(np.mean(std_dev) / 5)) # Increase Colors (less conserved quantization) = This will result in more clusters for K-Means.
    dynamic_threshold = int(np.mean(std_dev))


    return dynamic_threshold

def calculate_dynamic_threshold(image):
    pixels = image.reshape(-1, 3)
    std_dev = np.std(pixels, axis=0)
    dynamic_threshold = max(10, int(np.mean(std_dev) / 10))  # Scale for sensitivity
    return dynamic_threshold


# Superpixel Segmentation Function (SLIC)
def perform_superpixel_segmentation(image, n_segments=200):

    # Convert BGR image to LAB for better superpixel performance
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Apply SLIC for superpixel segmentation
    # More Conserved Quantization:
    # Increase compactness (e.g., compactness=20).
    # Less Conserved Quantization:
    # Decrease compactness (e.g., compactness=5).
    segments = slic(lab_image, n_segments=n_segments, compactness=10, start_label=0)
    # segments = slic(lab_image, n_segments=n_segments, compactness=10, start_label=0) # The compactness parameter in the SLIC algorithm controls how spatially compact the superpixels are.
                                                                                       # By increasing compactness, you can reduce the influence of colors far apart spatially.
    
    
    return segments

# K-Means Clustering Function
def perform_kmeans_clustering(image, num_clusters=10):
    # Flatten image to (num_pixels, 3)
    pixels = image.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # By default, K-Means initializes clusters randomly, which can lead to slightly different results.
    # For more consistent results, you can fix random_state or try different initialization methods:
    #kmeans = KMeans(n_clusters=num_clusters, random_state=42, init="k-means++")

    labels = kmeans.fit_predict(pixels)
    # Replace pixel colors with cluster centers
    quantized_image = kmeans.cluster_centers_.astype("uint8")[labels].reshape(image.shape)
    return quantized_image, kmeans.cluster_centers_

# Save Superpixel Visualization
def save_superpixel_visualization(image, segments, file_path):

    # Create a copy of the image to overlay boundaries
    overlay = image.copy()

    # Draw boundaries for superpixels
    for i in np.unique(segments):
        mask = segments == i
        overlay[mask] = overlay[mask].mean(axis=0)  # Assign mean color to region

    cv2.imwrite(file_path, overlay)

# Save Quantized Image
def save_quantized_image(image, file_path):
    cv2.imwrite(file_path, image)

def plot_regions_with_annotations(clustered_pixels, cluster_name, output_folder):

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    unique_colors = np.unique(clustered_pixels.reshape(-1, 3), axis=0)

    # Prepare grid for visualization
    grid_size = int(np.ceil(np.sqrt(len(unique_colors))))
    square_size = 100  # Size of each square in pixels
    canvas_size = (square_size * grid_size, square_size * grid_size, 3)
    canvas = np.ones(canvas_size, dtype=np.uint8) * 255  # White background

    # Draw each color as a square
    for idx, color in enumerate(unique_colors):

        x = (idx % grid_size) * square_size
        y = (idx // grid_size) * square_size
        canvas[y:y+square_size, x:x+square_size] = color

        # Add text annotation for RGB, CMYK, HSV
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
        c, m, y, k = rgb_to_cmyk(color[0], color[1], color[2])
        ax.text(
            x + square_size // 2, 
            y + square_size // 2 + 10, 
            f"RGB: {tuple(color)}\nCMYK: ({c:.2f}, {m:.2f}, {y:.2f}, {k:.2f})\nHSV: {tuple(hsv_color)}", 
            fontsize=6, 
            ha="center", 
            color="black",
            backgroundcolor="white"
        )

    ax.imshow(canvas)
    ax.axis("off")
    plt.title(f"Regions for {cluster_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"regions_{cluster_name}.png"))
    plt.close()

def rgb_to_cmyk(r, g, b):
    if r == 0 and g == 0 and b == 0:
        return 0, 0, 0, 1
    r_prime = r / 255
    g_prime = g / 255
    b_prime = b / 255
    k = 1 - max(r_prime, g_prime, b_prime)
    c = (1 - r_prime - k) / (1 - k)
    m = (1 - g_prime - k) / (1 - k)
    y = (1 - b_prime - k) / (1 - k)
    return c, m, y, k

def calculate_and_save_hsv_distributions(clustered_pixels, cluster_name, output_file):

    if clustered_pixels.size == 0:  # Skip empty clusters
        output_file.write(f"Cluster {cluster_name}: Empty\n")
        return

    # Flatten to (num_pixels, 3) HSV
    hsv_pixels = cv2.cvtColor(clustered_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    # Calculate mean and std for each channel
    hue_mean, hue_std = hsv_pixels[:, 0].mean(), hsv_pixels[:, 0].std()
    sat_mean, sat_std = hsv_pixels[:, 1].mean(), hsv_pixels[:, 1].std()
    val_mean, val_std = hsv_pixels[:, 2].mean(), hsv_pixels[:, 2].std()

    # Get the color's name
    color_name = closest_color_name(calculate_mean_rgb(clustered_pixels))

    # Save distribution to file
    output_file.write(
        f"Cluster: {cluster_name}\n"
        f"{color_name}: {{\"hue\": ({hue_mean:.2f}, {hue_std:.2f}), "
        f"\"saturation\": ({sat_mean:.2f}, {sat_std:.2f}), "
        f"\"value\": ({val_mean:.2f}, {val_std:.2f})}}\n"
    )

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

def calculate_mean_rgb(clustered_pixels):

    # Reshape to a 2D array (num_pixels, 3) for simplicity
    pixels = clustered_pixels.reshape(-1, 3)
    
    # Calculate mean along the 0th axis (across all pixels in the cluster)
    mean_rgb = np.mean(pixels, axis=0).astype(int)  # Ensure integer values for RGB
    return mean_rgb

# Iterate through kernel images
output_file = open(output_file_name, "w")
for file_name in os.listdir(input_folder):

    # Join file path
    file_path = os.path.join(input_folder, file_name)
    print(file_name)

    # If not an image, continue
    if os.path.splitext(file_path)[-1] not in [".png", ".jpg"]:
        continue

    # Load the image
    image = cv2.imread(file_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 1: Calculate dynamic threshold
    threshold = calculate_dynamic_threshold(image)
    print(threshold)

    # Step 2: Perform Superpixel Segmentation
    superpixel_segments = perform_superpixel_segmentation(image, n_segments = threshold * 10)
    # superpixel_segments = perform_superpixel_segmentation(image, n_segments=threshold * 2) # Fewer Superpixels (more conserved quantization)
    # superpixel_segments = perform_superpixel_segmentation(image, n_segments=threshold // 2) # More Superpixels (less conserved quantization)

    # Save Superpixel Visualization
    superpixel_path = os.path.join(output_folder, f"{file_name}_superpixels.png")
    save_superpixel_visualization(image, superpixel_segments, superpixel_path)

    # Step 3: Perform K-Means Clustering
    quantized_image, cluster_centers = perform_kmeans_clustering(image, num_clusters=threshold)
    # quantized_image, cluster_centers = perform_kmeans_clustering(image, num_clusters=8) # 8 colors = Force the number of clusters

    """
    Optional Post-Processing
    After quantization, you can further refine the image by merging similar clusters or applying filters:
    1. Merge clusters with similar centroids (color-wise).
    2. Apply median filtering to smooth out noise.

    # Calculate distances between cluster centers
    distances = cdist(cluster_centers, cluster_centers)

    # Merge clusters with small distances (e.g., <10 in RGB space)
    threshold = 10
    merged_centers = {}
    for i, center in enumerate(cluster_centers):
        for j, other_center in enumerate(cluster_centers):
            if i != j and distances[i, j] < threshold:
                merged_centers[j] = i  # Merge cluster j into i

    # Replace cluster assignments
    for old, new in merged_centers.items():
    labels[labels == old] = new
    """

    # Save Quantized Image
    quantized_path = os.path.join(output_folder, f"{file_name}_quantized.png")
    save_quantized_image(quantized_image, quantized_path)

    # Step 4: Write Results to Output File
    output_file.write(f"Processed {file_name}:\n")
    output_file.write(f"  - Superpixels saved to: {superpixel_path}\n")
    output_file.write(f"  - Quantized image saved to: {quantized_path}\n")
    output_file.write(f"  - Cluster Centers: {cluster_centers.tolist()}\n\n")

    # Iterate through cluster images
    unique_clusters = np.unique(quantized_image)  # Get unique cluster IDs
    for cluster_id in unique_clusters:

        # Cluster name will be the cluster_ID
        cluster_name = cluster_id
        cluster_mask = (quantized_image == cluster_id)
        clustered_pixels = rgb_image[cluster_mask]

        # Save annotated plots
        plot_regions_with_annotations(clustered_pixels, cluster_name, output_folder)

        # Save HSV distributions
        calculate_and_save_hsv_distributions(clustered_pixels, cluster_name, output_file)

# Closing file    
output_file.close()

