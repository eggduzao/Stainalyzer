
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

# Import
import os
import pandas as pd
import numpy as np


##################################
### Core Classes
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
### Core Functions
##################################


def load_cluster_data(file_path):
    """
    Load cluster data from a file into a Pandas DataFrame and construct HSV and RGB distribution objects.

    Args:
        file_path (str): Path to the input file containing cluster data.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed data with added HSVColorDistribution and RGBColorDistribution objects.
    """
    data = []
    current_image = None

    # Iterating on file
    with open(file_path, 'r') as file:
        for line in file:

            # Parsing new images
            line = line.strip()
            if line.startswith('#'):
                current_image = line[2:]  # Remove '# ' to get image name
            else:
                values = line.split('\t')
                
                # Parse RGB and HSV distributions
                rgb_distribution_image = RGBColorDistribution(
                    name=values[0],
                    red_mean=float(values[8]), red_std=float(values[9]),
                    green_mean=float(values[10]), green_std=float(values[11]),
                    blue_mean=float(values[12]), blue_std=float(values[13])
                )
                hsv_distribution_image = HSVColorDistribution(
                    name=values[0],
                    hue_mean=float(values[14]), hue_std=float(values[15]),
                    saturation_mean=float(values[16]), saturation_std=float(values[17]),
                    value_mean=float(values[18]), value_std=float(values[19])
                )
                rgb_distribution_quantized = RGBColorDistribution(
                    name=values[0],
                    red_mean=float(values[20]), red_std=float(values[21]),
                    green_mean=float(values[22]), green_std=float(values[23]),
                    blue_mean=float(values[24]), blue_std=float(values[25])
                )
                hsv_distribution_quantized = HSVColorDistribution(
                    name=values[0],
                    hue_mean=float(values[26]), hue_std=float(values[27]),
                    saturation_mean=float(values[28]), saturation_std=float(values[29]),
                    value_mean=float(values[30]), value_std=float(values[31])
                )

                # Append data to Panda's table
                data.append({
                    "image_name": current_image,
                    "color_name": values[0],
                    "total_pixels": int(values[1]),
                    "red_centroid": float(values[2]),
                    "green_centroid": float(values[3]),
                    "blue_centroid": float(values[4]),
                    "hue_centroid": float(values[5]),
                    "saturation_centroid": float(values[6]),
                    "value_centroid": float(values[7]),
                    "L_centroid": float(values[8]),
                    "A_centroid": float(values[9]),
                    "B_centroid": float(values[10]),
                    "rgb_distribution_image": rgb_distribution_image,
                    "hsv_distribution_image": hsv_distribution_image, 
                    "rgb_distribution_quantized": rgb_distribution_quantized,
                    "hsv_distribution_quantized": hsv_distribution_quantized 
                })

    return pd.DataFrame(data)

def iterate_images_and_colors(df):
    """
    Iterate over images and colors in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the cluster data.
    Yields:
        tuple: Image name and corresponding color information row by row.
    """
    for image_name, group in df.groupby("image_name"):
        for _, row in group.iterrows():
            yield image_name, row

##############################################################################################

def calculate_mahalanobis_distance(point, distribution):
    """
    Calculate the Mahalanobis distance between a point and a distribution.

    Args:
        point (tuple): The HSV point (H, S, V).
        distribution (HSVColorDistribution): HSV distribution.
    Returns:
        float: The Mahalanobis distance.
    """
    point_array = np.array(point)
    mean_array = np.array([distribution.m1, distribution.m2, distribution.m3])
    std_array = np.array([distribution.s1, distribution.s2, distribution.s3])
    return np.sqrt(np.sum(((point_array - mean_array) / std_array) ** 2))

def calculate_mahalanobis_distance_between_distributions(distribution1, distribution2):
    """
    Calculate the Mahalanobis distance between two HSV distributions.

    Args:
        distribution1 (HSVColorDistribution): First HSV distribution.
        distribution2 (HSVColorDistribution): Second HSV distribution.
    Returns:
        float: The Mahalanobis distance.
    """
    mean_diff = np.array([
        distribution1.m1 - distribution2.m1,
        distribution1.m2 - distribution2.m2,
        distribution1.m3 - distribution2.m3
    ])
    std_combined = np.array([
        max(np.sqrt(distribution1.s1 ** 2 + distribution2.s1 ** 2),1),
        max(np.sqrt(distribution1.s2 ** 2 + distribution2.s2 ** 2),1),
        max(np.sqrt(distribution1.s3 ** 2 + distribution2.s3 ** 2),1)
    ])
    return np.sqrt(np.sum((mean_diff / std_combined) ** 2))

##############################################################################################

def distance_centroid_RGB(centroid, dab_distribution):
    """
    Calculate a distance metric between an RGB centroid and the DAB-brown RGB distribution.

    Args:
        centroid (tuple): RGB values of the centroid (R, G, B).
        dab_distribution (RGBColorDistribution): DAB-brown RGB distribution.

    Returns:
        float: Calculated distance.
    """
    # Extract values from the distribution
    dab_mean = np.array([dab_distribution.red_mean, dab_distribution.green_mean, dab_distribution.blue_mean])
    dab_std = np.array([dab_distribution.red_std, dab_distribution.green_std, dab_distribution.blue_std])

    # Regularize standard deviation to avoid division by zero
    dab_std = np.maximum(dab_std, 1e-6)

    # Calculate weighted Euclidean distance
    diff = np.array(centroid) - dab_mean
    weighted_diff = diff / dab_std
    return np.sqrt(np.sum(weighted_diff ** 2))

def distance_centroid_HSV(centroid, dab_distribution):
    """
    Calculate a distance metric between an HSV centroid and the DAB-brown HSV distribution.

    Args:
        centroid (tuple): HSV values of the centroid (H, S, V) where H is in [0, 180].
        dab_distribution (HSVColorDistribution): DAB-brown HSV distribution.

    Returns:
        float: Calculated distance.
    """
    # Extract values from the distribution
    dab_mean = np.array([dab_distribution.hue_mean, dab_distribution.saturation_mean, dab_distribution.value_mean])
    dab_std = np.array([dab_distribution.hue_std, dab_distribution.saturation_std, dab_distribution.value_std])

    # Regularize standard deviation to avoid division by zero
    dab_std = np.maximum(dab_std, 1e-6)

    # Handle circular distance for Hue
    hue_diff = np.abs(centroid[0] - dab_mean[0])
    hue_diff = min(hue_diff, 180 - hue_diff)  # Circular difference for [0, 180]

    # Compute weighted Euclidean distance
    diff = np.array([hue_diff, centroid[1] - dab_mean[1], centroid[2] - dab_mean[2]])
    weighted_diff = diff / dab_std
    return np.sqrt(np.sum(weighted_diff ** 2))

def distance_distribution_RGB(dist1, dist2):
    """
    Calculate a distance metric between two RGB distributions.

    Args:
        dist1 (RGBColorDistribution): First RGB distribution.
        dist2 (RGBColorDistribution): Second RGB distribution.

    Returns:
        float: Calculated distance.
    """
    # Extract means and standard deviations
    mean_diff = np.array([dist1.red_mean - dist2.red_mean,
                          dist1.green_mean - dist2.green_mean,
                          dist1.blue_mean - dist2.blue_mean])
    std_combined = np.sqrt(np.array([
        dist1.red_std ** 2 + dist2.red_std ** 2,
        dist1.green_std ** 2 + dist2.green_std ** 2,
        dist1.blue_std ** 2 + dist2.blue_std ** 2
    ]))

    # Regularize combined standard deviation
    std_combined = np.maximum(std_combined, 1e-6)

    # Compute Mahalanobis-like distance
    weighted_diff = mean_diff / std_combined
    return np.sqrt(np.sum(weighted_diff ** 2))

def distance_distribution_HSV(dist1, dist2):
    """
    Calculate a distance metric between two HSV distributions.

    Args:
        dist1 (HSVColorDistribution): First HSV distribution.
        dist2 (HSVColorDistribution): Second HSV distribution.

    Returns:
        float: Calculated distance.
    """
    # Extract means and standard deviations
    mean_diff = np.array([
        np.abs(dist1.hue_mean - dist2.hue_mean),
        dist1.saturation_mean - dist2.saturation_mean,
        dist1.value_mean - dist2.value_mean
    ])

    # Handle circular distance for Hue
    mean_diff[0] = min(mean_diff[0], 180 - mean_diff[0])  # Circular difference for [0, 180]

    # Combine standard deviations
    std_combined = np.sqrt(np.array([
        dist1.hue_std ** 2 + dist2.hue_std ** 2,
        dist1.saturation_std ** 2 + dist2.saturation_std ** 2,
        dist1.value_std ** 2 + dist2.value_std ** 2
    ]))

    # Regularize combined standard deviation
    std_combined = np.maximum(std_combined, 1e-6)

    # Compute Mahalanobis-like distance
    weighted_diff = mean_diff / std_combined
    return np.sqrt(np.sum(weighted_diff ** 2))

##############################################################################################

def main():


# Load the data
df = load_cluster_data(input_color_cluster_file_name)

# Output file
output_summary_file_name = os.path.join(output_location, output_results_file_name)
output_summary_file = open(output_summary_file_name, "w")
output_vectors_file_name = os.path.join(output_location, output_vectors_file_name)
output_vectors_file = open(output_vectors_file_name, "w")

# Define the standard DAB-brown distribution in RGB, HSV and LAB spaces
dab_brown_distribution_rgb = RGBColorDistribution(
    name="DAB Brown",
    red_mean=140, red_std=20,
    green_mean=110, green_std=20,
    blue_mean=90, blue_std=20
)
dab_brown_distribution_hsv = HSVColorDistribution(
    name="DAB Brown",
    hue_mean=20, hue_std=5,
    saturation_mean=100, saturation_std=10,
    value_mean=80, value_std=10
)
dab_brown_distribution_lab = LABColorDistribution(
    name="DAB Brown",
    l_mean=50, l_std=10,    # Lightness component
    a_mean=10, a_std=5,     # Green-red axis component
    b_mean=20, b_std=5      # Blue-yellow axis component
)

# Iterate through images and colors
for image_name, row in iterate_images_and_colors(df):

    # Cluster color centroid RGB
    centroid_rgb = (row["red_centroid"], row["green_centroid"], row["blue_centroid"])
    centroid_hsv = (row["hue_centroid"], row["saturation_centroid"], row["value_centroid"])

    # Cluster HSV distribution
    rgb_distribution_image = row["rgb_distribution_image"]
    hsv_distribution_image = row["hsv_distribution_image"]
    rgb_distribution_quantized = row["rgb_distribution_quantized"]
    hsv_distribution_quantized = row["hsv_distribution_quantized"]

    # Calculate Mahalanobis distances for centroids
    centroid_distance_rgb = calculate_mahalanobis_distance(centroid_rgb, dab_brown_distribution_rgb)
    centroid_distance_hsv = calculate_mahalanobis_distance(centroid_hsv, dab_brown_distribution_hsv)
    centroid_distance_rgb2 = distance_centroid_RGB(centroid_rgb, dab_brown_distribution_rgb)
    centroid_distance_hsv2 = distance_centroid_HSV(centroid_hsv, dab_brown_distribution_hsv)

    # Calculate Mahalanobis distances for distributions
    distribution_distance_rgb_image = calculate_mahalanobis_distance_between_distributions(rgb_distribution_image, dab_brown_distribution_rgb)
    distribution_distance_rgb_quantized = calculate_mahalanobis_distance_between_distributions(rgb_distribution_quantized, dab_brown_distribution_rgb)
    distribution_distance_hsv_image = calculate_mahalanobis_distance_between_distributions(hsv_distribution_image, dab_brown_distribution_hsv)
    distribution_distance_hsv_quantized = calculate_mahalanobis_distance_between_distributions(hsv_distribution_quantized, dab_brown_distribution_hsv)
    distribution_distance_rgb_image2 = distance_distribution_RGB(rgb_distribution_image, dab_brown_distribution_rgb)
    distribution_distance_rgb_quantized2 = distance_distribution_RGB(rgb_distribution_quantized, dab_brown_distribution_rgb)
    distribution_distance_hsv_image2 = distance_distribution_HSV(hsv_distribution_image, dab_brown_distribution_hsv)
    distribution_distance_hsv_quantized2 = distance_distribution_HSV(hsv_distribution_quantized, dab_brown_distribution_hsv)

    # Write results to output file

    output_summary_file.write(f"# Image\t{row['image_name']}\tColor\t{row['color_name']}\tTotal Pixels\t{row['total_pixels']}:\n")
    output_vectors_file.write(f"{row['color_name']}\t")
    output_vectors_file.write(f"{centroid_distance_rgb2:.2f}\t")
    output_vectors_file.write(f"{distribution_distance_rgb_image2:.2f}\t")
    output_vectors_file.write(f"{distribution_distance_rgb_quantized2:.2f}\t")
    output_vectors_file.write(f"{centroid_distance_hsv2:.2f}\t")
    output_vectors_file.write(f"{distribution_distance_hsv_image2:.2f}\t")
    output_vectors_file.write(f"{distribution_distance_hsv_quantized2:.2f}\n")

# Closing file
output_summary_file.close()
output_vectors_file.close()

if __name__ == "__main__":

    # Main Script
    input_color_cluster_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining8/results.txt"
    output_location = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/supercolider4/"
    output_results_file_name = "results.txt"
    output_vectors_file_name = "vectors.txt"
    output_plot1_prefix = "barplot"

    # Main function
    main(input_color_cluster_file_name, output_location, output_results_file_name, output_vectors_file_name, output_plot1_prefix)

"""
    # Cluster color centroid RGB
    centroid_rgb = (row["red_centroid"], row["green_centroid"], row["blue_centroid"])
    centroid_hsv = (row["hue_centroid"], row["saturation_centroid"], row["value_centroid"])

    # Cluster HSV distribution
    rgb_distribution_image = row["rgb_distribution_image"]
    hsv_distribution_image = row["hsv_distribution_image"]
    rgb_distribution_quantized = row["rgb_distribution_quantized"]
    hsv_distribution_quantized = row["hsv_distribution_quantized"]

    # Calculate Mahalanobis distances for centroids
    centroid_distance_rgb = calculate_mahalanobis_distance(centroid_rgb, dab_brown_distribution_rgb)
    centroid_distance_hsv = calculate_mahalanobis_distance(centroid_hsv, dab_brown_distribution_hsv)
    centroid_distance_rgb2 = distance_centroid_RGB(centroid_rgb, dab_brown_distribution_rgb)
    centroid_distance_hsv2 = distance_centroid_HSV(centroid_hsv, dab_brown_distribution_hsv)

    # Calculate Mahalanobis distances for distributions
    distribution_distance_rgb_image = calculate_mahalanobis_distance_between_distributions(rgb_distribution_image, dab_brown_distribution_rgb)
    distribution_distance_rgb_quantized = calculate_mahalanobis_distance_between_distributions(rgb_distribution_quantized, dab_brown_distribution_rgb)
    distribution_distance_hsv_image = calculate_mahalanobis_distance_between_distributions(hsv_distribution_image, dab_brown_distribution_hsv)
    distribution_distance_hsv_quantized = calculate_mahalanobis_distance_between_distributions(hsv_distribution_quantized, dab_brown_distribution_hsv)
    distribution_distance_rgb_image2 = distance_distribution_RGB(rgb_distribution_image, dab_brown_distribution_rgb)
    distribution_distance_rgb_quantized2 = distance_distribution_RGB(rgb_distribution_quantized, dab_brown_distribution_rgb)
    distribution_distance_hsv_image2 = distance_distribution_HSV(hsv_distribution_image, dab_brown_distribution_hsv)
    distribution_distance_hsv_quantized2 = distance_distribution_HSV(hsv_distribution_quantized, dab_brown_distribution_hsv)
"""



