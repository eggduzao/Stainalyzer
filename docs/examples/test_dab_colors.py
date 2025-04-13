
import os
import re
import cv2
import numpy as np

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

    print(f"Checking cluster: Mean HSV {mean_hsv}, Std HSV {std_hsv}")
    print(f"Mean range check: {h_range_adj}, {s_range_adj}, {v_range_adj}")
    print(f"Std range check: {h_std_range}, {s_std_range}, {v_std_range}")

    return mean_in_range and std_in_range

# Main Script
input_color_cluster_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining5/color_cluster.txt"

# Reading the image parameters
image_entries = read_color_cluster(input_color_cluster_file_name)

for image in image_entries:

    # Image
    image_without_extension = os.path.splitext(image[0])[0]

    # Precompute KDEs
    #cluster_kdes = precompute_kdes(image[1])

    # Determine DAB-positive clusters
    dab_clusters = {
        name: params for name, params in image[1].items()
        if is_dab_like(params[:3], params[3:], (10, 30), (40, 80), (20, 70), 0, 0, 0, 0, 0, 0)
    }

    for name, params in image[1].items():
        print(f"Cluster {name}: Mean HSV = {params[:3]}, Std HSV = {params[3:]}")

    for name, params in dab_clusters.items():
        print(f"{name}: {params}")

