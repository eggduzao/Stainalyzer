
######################################################################################################
# Contagem de Pixels
#   3752-20     913916
#   7716-20G    798591
#   8414-20F    0
#   3833-21H    0
#   5086-21F    194889
#   5829-21B    479505
#   6520-21E    815705
######################################################################################################

import os
import re
import cv2
import numpy as np
from scipy.stats import gaussian_kde
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

def read_color_cluster(file_path):
    image_entries = []
    current_image = None
    current_colors = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):  # Indicates a new image entry
                if current_image and current_colors:  # Save the previous entry
                    image_entries.append((current_image, current_colors))
                current_image = line[2:].strip()  # Get the image name (after "# ")
                current_colors = {}
            elif line:  # Parse the color cluster details
                parts = line.split("\t")
                if len(parts) == 7:
                    color_name = parts[0]
                    hsv_params = list(map(float, parts[1:]))
                    current_colors[color_name] = hsv_params
        
        # Add the last entry to the list
        if current_image and current_colors:
            image_entries.append((current_image, current_colors))
    
    return image_entries

def is_dab_like(mean_hsv, std_hsv, h_range, s_range, v_range, h_mean_threshold=5, s_mean_threshold=5, v_mean_threshold=5, h_std_threshold=2, s_std_threshold=2, v_std_threshold=2):
    mean_h, mean_s, mean_v = mean_hsv
    std_h, std_s, std_v = std_hsv

    h_range_adj = (h_range[0] - h_mean_threshold, h_range[1] + h_mean_threshold)
    s_range_adj = (s_range[0] - s_mean_threshold, s_range[1] + s_mean_threshold)
    v_range_adj = (v_range[0] - v_mean_threshold, v_range[1] + v_mean_threshold)

    h_std_range = (h_range[0] - h_std_threshold, h_range[1] + h_std_threshold)
    s_std_range = (s_range[0] - s_std_threshold, s_range[1] + s_std_threshold)
    v_std_range = (v_range[0] - v_std_threshold, v_range[1] + v_std_threshold)

    mean_in_range = (h_range_adj[0] <= mean_h <= h_range_adj[1]) and \
                    (s_range_adj[0] <= mean_s <= s_range_adj[1]) and \
                    (v_range_adj[0] <= mean_v <= v_range_adj[1])

    std_in_range = (h_std_range[0] <= std_h <= h_std_range[1]) and \
                   (s_std_range[0] <= std_s <= s_std_range[1]) and \
                   (v_std_range[0] <= std_v <= v_std_range[1])

    return mean_in_range and std_in_range

def precompute_kdes(color_clusters):
    """
    Precompute KDEs for each cluster to avoid redundant calculations.
    """
    cluster_kdes = {}
    for name, params in color_clusters.items():
        mean_hsv = np.array(params[:3])
        std_hsv = np.array(params[3:])
        samples = np.random.normal(loc=mean_hsv, scale=std_hsv, size=(1000, 3))
        kde = gaussian_kde(samples.T)
        cluster_kdes[name] = kde
    return cluster_kdes

def process_pixel(pixel_hsv, cluster_kdes, dab_clusters):
    """
    Calculate probabilities for a single pixel using precomputed KDEs.
    """
    probabilities = []
    for name, kde in cluster_kdes.items():
        probabilities.append(kde(pixel_hsv))
    max_prob_index = np.argmax(probabilities)
    max_prob_cluster = list(cluster_kdes.keys())[max_prob_index]
    is_dab = max_prob_cluster in dab_clusters
    return probabilities, is_dab

def classify_pixels_parallel(image_hsv, cluster_kdes, dab_clusters, max_workers=4):
    """
    Classify pixels in parallel using ProcessPoolExecutor.
    """
    height, width, _ = image_hsv.shape
    pixels = image_hsv.reshape(-1, 3)  # Flatten image into a list of pixels

    # Parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda pixel: process_pixel(pixel, cluster_kdes, dab_clusters), pixels
        ))

    # Reshape results back to image dimensions
    pixel_probabilities = np.array([res[0] for res in results]).reshape(height, width, -1)
    dab_pixels = sum(res[1] for res in results)
    return pixel_probabilities, dab_pixels

def write_summary(output_file, image_name, dab_pixels, total_pixels):
    percentage_dab = (dab_pixels / total_pixels) * 100
    output_file.write(f"{image_name}\t{dab_pixels}\t{percentage_dab:.2f}%\n")

def generate_violin_plots(image_name, pixel_probabilities, color_clusters, dab_clusters, output_location, prefix):
    channels = ['H', 'S', 'V']
    num_clusters = len(color_clusters)

    for i, channel in enumerate(channels):
        plt.figure(figsize=(12, 8))

        # Extract channel-specific probabilities
        channel_probs = pixel_probabilities[:, :, i]  # Correct slicing
        cluster_probs = channel_probs.reshape(-1, num_clusters)

        # Create violins
        sns.violinplot(data=cluster_probs, inner="box")

        # Annotate with cluster means
        for idx, (color_name, params) in enumerate(color_clusters.items()):
            plt.scatter([idx], [params[i]], color='red', label=f"Mean ({color_name})")

        # Add labels and save plot
        plt.title(f"{channel} Violin Plot for {image_name}")
        plt.xlabel("Clusters")
        plt.ylabel("Probability")
        output_file = os.path.join(output_location, f"{prefix}_{image_name}_{channel}.png")
        plt.savefig(output_file)
        plt.close()

# Main Script
input_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/"
output_location = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/supercolider2/"
output_summary_file_name = "results.txt"
output_super_plot1_prefix = "superplot1"
output_super_plot2_prefix = "superplot2"
input_color_cluster_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining5/color_cluster.txt"

# Reading the image parameters
image_entries = read_color_cluster(input_color_cluster_file_name)

# Output file
output_summary_file = open(output_summary_file_name, "w")

for image in image_entries:
    image_without_extension = os.path.splitext(image[0])[0]
    image_file_name = os.path.join(input_file_name, image[0])
    image_bgr = cv2.imread(image_file_name)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Precompute KDEs
    #cluster_kdes = precompute_kdes(image[1])

    # Determine DAB-positive clusters
    dab_clusters = {
        name: params for name, params in image[1].items()
        if is_dab_like(params[:3], params[3:], (10, 30), (40, 80), (20, 70))
    }

    for name, params in dab_clusters:
        print(f"{name}: params")

    # Classify pixels in parallel
    #pixel_probabilities, dab_pixels = classify_pixels_parallel(image_hsv, cluster_kdes, dab_clusters, max_workers=4)

    # Write summary
    #total_pixels = image_hsv.shape[0] * image_hsv.shape[1]
    #write_summary(output_summary_file, image_without_extension, dab_pixels, total_pixels)

    # Generate violin plots (existing function)
    #generate_violin_plots(image_without_extension, pixel_probabilities, image[1], dab_clusters, output_location, output_super_plot1_prefix)

output_summary_file.close()
"""


def classify_pixels(image_hsv, color_clusters, dab_clusters):
    height, width, _ = image_hsv.shape
    pixel_probabilities = np.zeros((height, width, len(color_clusters)))
    dab_pixels = 0

    cluster_kdes = {}
    for color_name, cluster_data in color_clusters.items():
        mean_hsv = np.array(cluster_data[:3])
        std_hsv = np.array(cluster_data[3:])
        samples = np.random.normal(loc=mean_hsv, scale=std_hsv, size=(1000, 3))
        kde = gaussian_kde(samples.T)
        cluster_kdes[color_name] = kde

    for y in range(height):
        for x in range(width):
            pixel_hsv = image_hsv[y, x]
            probabilities = [
                kde(pixel_hsv)[0] for kde in cluster_kdes.values()
            ]
            pixel_probabilities[y, x] = probabilities
            max_prob_index = np.argmax(probabilities)
            max_prob_cluster = list(color_clusters.keys())[max_prob_index]
            if max_prob_cluster in dab_clusters:
                dab_pixels += 1

    return pixel_probabilities, dab_pixels



import os
import re

def read_color_cluster(file_path):
    image_entries = []
    current_image = None
    current_colors = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):  # Indicates a new image entry
                if current_image and current_colors:  # Save the previous entry
                    image_entries.append((current_image, current_colors))
                current_image = line[2:].strip()  # Get the image name (after "# ")
                current_colors = {}
            elif line:  # Parse the color cluster details
                parts = line.split("\t")
                if len(parts) == 7:
                    color_name = parts[0]
                    hsv_params = list(map(float, parts[1:]))
                    current_colors[color_name] = hsv_params
        
        # Add the last entry to the list
        if current_image and current_colors:
            image_entries.append((current_image, current_colors))
    
    return image_entries

# PLEASE PUT FUNCTIONS HERE

# Input files
input_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/"
output_location = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/supercolider2/"
output_summary_file_name = "results.txt"
output_super_plot1_prefix = "superplot1"
output_super_plot2_prefix = "superplot2"
input_color_cluster_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining5/color_cluster.txt"

# Reading the image parameters
image_entries = read_color_cluster(input_color_cluster_file_name)

# Output file
output_summary_file = open(output_summary_file_name, "w")

# Print the output for verification
for image in image_entries:

    # Image file name
    image_without_extension = image.split(".")[0]
    image_file_name = os.path.join(input_file_name, image[0])

    # 1 Part - Read Image as RGB and HSV

    for color_name, parameter_vector in image[1].items():

        # Part 2:

        # 2.1. - Classifying each color as 'DAB-positive' or 'DAB-negative

        # 2.2. - Create Gaussian for posterior probability calculation

        # 2.3. - Iterate in the pixels of the image, storing and counting the DAB-positives

    # Part 3: Write the results - for this image
    output_summary_file.write(#WRITE HERE)

    # Part 4: Plots

    # 4.1. - Make the Plot 1 with all os violins in the violin plot
    #output_super_plot1_path = os.path.join(output_location, output_super_plot1_prefix)
    #output_super_plot1_file_name_H = f"{output_super_plot1_path}_{image_without_extension}_H.png"
    #output_super_plot1_file_name_S = f"{output_super_plot1_path}_{image_without_extension}_S.png"
    #output_super_plot1_file_name_V = f"{output_super_plot1_path}_{image_without_extension}_V.png"

    # 4.2. - Make the Plot 2 with 2 violins in the violin plots
    #output_super_plot2_path = os.path.join(output_location, output_super_plot2_prefix)
    #output_super_plot2_file_name_H = f"{output_super_plot2_path}_{image_without_extension}_H.png"
    #output_super_plot2_file_name_S = f"{output_super_plot2_path}_{image_without_extension}_S.png"
    #output_super_plot2_file_name_V = f"{output_super_plot2_path}_{image_without_extension}_V.png"

# Closing file
output_summary_file.close()
"""

"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, ks_2samp
from statsmodels.stats.multitest import multipletests

# Input and output
input_file = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining5/color_cluster.txt"
input_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/"
output_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/supercolider2/"
os.makedirs(output_folder, exist_ok=True)

# Thresholds for DAB classification
h_mean_range = (10, 30)
s_mean_range = (40, 80)
v_mean_range = (20, 70)
h_mean_threshold = 5
s_mean_threshold = 10
v_mean_threshold = 10
h_std_threshold = 5
s_std_threshold = 10
v_std_threshold = 10

# Adjusted thresholds
h_mean_range = (h_mean_range[0] - h_mean_threshold, h_mean_range[1] + h_mean_threshold)
s_mean_range = (s_mean_range[0] - s_mean_threshold, s_mean_range[1] + s_mean_threshold)
v_mean_range = (v_mean_range[0] - v_mean_threshold, v_mean_range[1] + v_mean_threshold)
h_std_range = (h_mean_range[0] - h_std_threshold, h_mean_range[1] + h_std_threshold)
s_std_range = (s_mean_range[0] - s_std_threshold, s_mean_range[1] + s_std_threshold)
v_std_range = (v_mean_range[0] - v_std_threshold, v_mean_range[1] + v_std_threshold)

# Helper function: Check if cluster is DAB-like
def is_dab_like(h_mean, h_std, s_mean, s_std, v_mean, v_std):
    mean_in_range = (
        h_mean_range[0] <= h_mean <= h_mean_range[1] and
        s_mean_range[0] <= s_mean <= s_mean_range[1] and
        v_mean_range[0] <= v_mean <= v_mean_range[1]
    )
    std_in_range = (
        h_std_range[0] <= h_std <= h_std_range[1] and
        s_std_range[0] <= s_std <= s_std_range[1] and
        v_std_range[0] <= v_std <= v_std_range[1]
    )
    return mean_in_range and std_in_range

# Read cluster data
with open(input_file, 'r') as f:
    lines = f.readlines()


### Check entries - Update to correct the entry

entries = []
current_image = None
for line in lines:
    line = line.strip()
    if line.startswith('#'):
        current_image = line[1:].strip()
    elif current_image:
        entries.append((current_image, line.split('\t')))

print(entries)

# Initialize results
results = []
dab_pixels_summary = []

for image_name, clusters in entries:

    image_path = os.path.join(input_folder, image_name)
    if not os.path.exists(image_path):
        print(f"Image '{image_name}' not found in {input_folder}. Skipping...")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Image '{image_name}' could not be loaded. Skipping...")
        continue

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
    dab_pixels = 0

    # Parse clusters and classify as DAB-stain or not
    cluster_info = []
    for cluster_data in clusters:
        color_name, h_mean, h_std, s_mean, s_std, v_mean, v_std = cluster_data.split()
        h_mean, h_std = float(h_mean), float(h_std)
        s_mean, s_std = float(s_mean), float(s_std)
        v_mean, v_std = float(v_mean), float(v_std)

        is_dab = is_dab_like(h_mean, h_std, s_mean, s_std, v_mean, v_std)
        cluster_info.append({
            'color_name': color_name,
            'h_mean': h_mean,
            'h_std': h_std,
            's_mean': s_mean,
            's_std': s_std,
            'v_mean': v_mean,
            'v_std': v_std,
            'is_dab': is_dab
        })

    # Calculate posterior probabilities
    posterior_probs = np.zeros((hsv_image.shape[0], hsv_image.shape[1], len(cluster_info)))

    for i, cluster in enumerate(cluster_info):
        cluster_h_mean = cluster['h_mean']
        cluster_s_mean = cluster['s_mean']
        cluster_v_mean = cluster['v_mean']

        cluster_h_std = cluster['h_std']
        cluster_s_std = cluster['s_std']
        cluster_v_std = cluster['v_std']

        # Gaussian KDE for probability estimation
        kernel_h = gaussian_kde([cluster_h_mean, cluster_h_std])
        kernel_s = gaussian_kde([cluster_s_mean, cluster_s_std])
        kernel_v = gaussian_kde([cluster_v_mean, cluster_v_std])

        # Calculate probabilities
        h_probs = kernel_h(hsv_image[:, :, 0].flatten())
        s_probs = kernel_s(hsv_image[:, :, 1].flatten())
        v_probs = kernel_v(hsv_image[:, :, 2].flatten())

        # Reshape probabilities back into image shape
        h_probs = h_probs.reshape(hsv_image.shape[0], hsv_image.shape[1])
        s_probs = s_probs.reshape(hsv_image.shape[0], hsv_image.shape[1])
        v_probs = v_probs.reshape(hsv_image.shape[0], hsv_image.shape[1])

        # Posterior probability as product of channel probabilities
        posterior_probs[:, :, i] = h_probs * s_probs * v_probs

    # Classify pixels
    max_probs = np.argmax(posterior_probs, axis=2)
    for i, cluster in enumerate(cluster_info):
        if cluster['is_dab']:
            dab_pixels += np.sum(max_probs == i)

    dab_pixels_summary.append({
        'image_name': image_name,
        'dab_pixels': dab_pixels,
        'dab_ratio': dab_pixels / total_pixels
    })

    # Generate violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=posterior_probs.reshape(-1, len(cluster_info)))
    plt.savefig(os.path.join(output_folder, f"{image_name}_violin_plot.png"))
    plt.close()

    ### Check plots - Remove above

    # Generate Final Plot 1: HSV Distributions for all clusters
    plt.figure(figsize=(12, 8))
    for i, cluster in enumerate(cluster_info):
        cluster_data = hsv_image[max_probs == i].reshape(-1, 3)
        if cluster_data.size == 0:
            continue
        sns.violinplot(data=cluster_data, split=True, inner="quartile")
        plt.title(f"Cluster HSV Distribution for {figure_name}")
        plt.savefig(os.path.join(output_folder, f"{figure_name}_cluster_hsv_plot.png"))
        plt.close()

    # Generate Final Plot 2: DAB-positive vs. DAB-negative
    dab_data = hsv_image[max_probs == [i for i, c in enumerate(cluster_info) if c['is_dab']]].reshape(-1, 3)
    non_dab_data = hsv_image[max_probs == [i for i, c in enumerate(cluster_info) if not c['is_dab']]].reshape(-1, 3)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[dab_data, non_dab_data], split=True, inner="quartile")
    plt.title(f"DAB-positive vs. DAB-negative Distribution for {figure_name}")
    plt.savefig(os.path.join(output_folder, f"{figure_name}_dab_vs_nondab_plot.png"))
    plt.close()


# Save dab pixel summary
summary_df = pd.DataFrame(dab_pixels_summary)
summary_df.to_csv(os.path.join(output_folder, "dab_pixel_summary.txt"), sep='\t', index=False)

# Who is output file?
output_file.close()

"""

"""
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to determine if a cluster is DAB-stain
def is_dab_like(h_mean, h_std, s_mean, s_std, v_mean, v_std, h_thresh=(10, 30), s_thresh=(40, 80), v_thresh=(20, 70)):
    h_in_range = h_thresh[0] <= h_mean <= h_thresh[1]
    s_in_range = s_thresh[0] <= s_mean <= s_thresh[1]
    v_in_range = v_thresh[0] <= v_mean <= v_thresh[1]
    return h_in_range and s_in_range and v_in_range

# Function to calculate Gaussian posterior probability for a pixel
def calculate_posterior_probability(hsv_pixel, h_mean, h_std, s_mean, s_std, v_mean, v_std):
    h, s, v = hsv_pixel
    h_prob = np.exp(-0.5 * ((h - h_mean) / h_std) ** 2) / (np.sqrt(2 * np.pi) * h_std)
    s_prob = np.exp(-0.5 * ((s - s_mean) / s_std) ** 2) / (np.sqrt(2 * np.pi) * s_std)
    v_prob = np.exp(-0.5 * ((v - v_mean) / v_std) ** 2) / (np.sqrt(2 * np.pi) * v_std)
    return h_prob * s_prob * v_prob

# Entry structure
entries = [
    ("Figure1.jpg", [['colorname11', 15, 5, 60, 10, 40, 8], ['colorname12', 25, 6, 50, 12, 30, 7]]),
    ("Figure2.jpg", [['colorname21', 18, 4, 55, 9, 35, 6], ['colorname22', 22, 5, 48, 10, 32, 8]])
]

output_folder = "./output/"
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, "dab_pixel_summary.txt")
output_file = open(output_file_path, "w")

# Process entries
for figure_name, cluster_vec in entries:
    figure_path = f"./path_to_images/{figure_name}"  # Replace with the correct path
    if not os.path.exists(figure_path):
        print(f"Image {figure_name} not found. Skipping...")
        continue

    # Load and preprocess image
    image = cv2.imread(figure_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]

    cluster_info = []
    posterior_probs = np.zeros((hsv_image.shape[0], hsv_image.shape[1], len(cluster_vec)))
    dab_pixels = 0

    # Process clusters
    for i, cluster in enumerate(cluster_vec):
        color_name, h_mean, h_std, s_mean, s_std, v_mean, v_std = cluster
        is_dab = is_dab_like(h_mean, h_std, s_mean, s_std, v_mean, v_std)
        cluster_info.append({'color_name': color_name, 'is_dab': is_dab})

        # Calculate posterior probability for each pixel
        for x in range(hsv_image.shape[0]):
            for y in range(hsv_image.shape[1]):
                pixel_hsv = hsv_image[x, y]
                posterior_probs[x, y, i] = calculate_posterior_probability(
                    pixel_hsv, h_mean, h_std, s_mean, s_std, v_mean, v_std
                )

    # Classify pixels based on maximum posterior probability
    max_probs = np.argmax(posterior_probs, axis=2)
    for i, cluster in enumerate(cluster_info):
        if cluster['is_dab']:
            dab_pixels += np.sum(max_probs == i)

    # Save results
    dab_ratio = dab_pixels / total_pixels
    output_file.write(f"{figure_name}\t{dab_pixels}\t{dab_ratio}\t{total_pixels}\n")

    # Generate Final Plot 1: HSV Distributions for all clusters
    plt.figure(figsize=(12, 8))
    for i, cluster in enumerate(cluster_info):
        cluster_data = hsv_image[max_probs == i].reshape(-1, 3)
        if cluster_data.size == 0:
            continue
        sns.violinplot(data=cluster_data, split=True, inner="quartile")
        plt.title(f"Cluster HSV Distribution for {figure_name}")
        plt.savefig(os.path.join(output_folder, f"{figure_name}_cluster_hsv_plot.png"))
        plt.close()

    # Generate Final Plot 2: DAB-positive vs. DAB-negative
    dab_data = hsv_image[max_probs == [i for i, c in enumerate(cluster_info) if c['is_dab']]].reshape(-1, 3)
    non_dab_data = hsv_image[max_probs == [i for i, c in enumerate(cluster_info) if not c['is_dab']]].reshape(-1, 3)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[dab_data, non_dab_data], split=True, inner="quartile")
    plt.title(f"DAB-positive vs. DAB-negative Distribution for {figure_name}")
    plt.savefig(os.path.join(output_folder, f"{figure_name}_dab_vs_nondab_plot.png"))
    plt.close()

output_file.close()
"""
