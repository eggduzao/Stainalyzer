
# Import
import os
import cv2
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

# Define HSV Color Distribution Class
class HSVColorDistribution:
    def __init__(self, name, mean_h, std_h, mean_s, std_s, mean_v, std_v):
        self.name = name
        self.mean = np.array([mean_h, mean_s, mean_v])
        self.covariance = np.diag([std_h**2, std_s**2, std_v**2])

# Read Color Cluster File
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

# Define Mahalanobis Distance Function
def calculate_mahalanobis_distance(distribution1, distribution2):
    diff = distribution1.mean - distribution2.mean
    inv_cov = np.linalg.inv(distribution2.covariance)
    distance = np.sqrt(diff.T @ inv_cov @ diff)
    return distance

# Main Script
input_color_cluster_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining6/color_cluster.txt"
output_location = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/supercolider3/"
output_results_file_name = "results.txt"
output_plot1_prefix = "barplot"

# Reading the image parameters
image_entries = read_color_cluster(input_color_cluster_file_name)

# Output file
output_summary_file_name = os.join(output_location, output_distances_file_name)
output_summary_file = open(output_summary_file_name, "w")

# Create the "Gold-standard" Mahalanobis distribution for DAB-Brown
dab_brown = HSVColorDistribution(
    name="DAB-Brown",
    mean_h=30, std_h=10,
    mean_s=100, std_s=20,
    mean_v=150, std_v=30
) # Example values, adjust as necessary

# Prepare data for plotting
plot_data = []

# Iterating through every image
for image in image_entries:
    # Get object names
    image_name = image[0]  # Contains the image file name with extension
    color_distributions = image[1]  # Dictionary of color distributions

    # Iterate through color distributions
    for color_name, hsv_params in color_distributions.items():
        # Create HSVColorDistribution object
        color_distribution = HSVColorDistribution(
            name=color_name,
            mean_h=hsv_params[0], std_h=hsv_params[1],
            mean_s=hsv_params[2], std_s=hsv_params[3],
            mean_v=hsv_params[4], std_v=hsv_params[5]
        )

        # Calculate Mahalanobis distance to DAB-Brown
        distance = calculate_mahalanobis_distance(color_distribution, dab_brown)

        # Write distance to output file
        output_summary_file.write(f"{color_name}\t{distance}\n")

        # Append data for plotting
        plot_data.append((color_name, distance))

# Close summary file name
output_summary_file.close()

# Generate bar plot
color_names = [item[0] for item in plot_data]
distances = [item[1] for item in plot_data]

plt.figure(figsize=(10, 6))
plt.bar(color_names, distances)
plt.yscale('log')  # Use log scale for better visualization
plt.xlabel("Colors (Webcolor Name)")
plt.ylabel("Mahalanobis Distance (Log Scale)")
plt.title("Mahalanobis Distance of Colors to DAB-Brown")
plt.xticks(rotation=90)
plt.tight_layout()

# Save plot
output_plot1_path = os.path.join(output_location, f"{output_plot1_prefix}.png")
plt.savefig(output_plot1_path)
plt.show()

