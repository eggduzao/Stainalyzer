
##################################
### Import
##################################

# Import
import os
import pandas as pd
import numpy as np
from math import sqrt, pi, cos, sin
from scipy.stats import wasserstein_distance

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
            self.l_mean, self.l_std,
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
                    red_mean=float(values[11]), red_std=float(values[14]),
                    green_mean=float(values[12]), green_std=float(values[15]),
                    blue_mean=float(values[13]), blue_std=float(values[16])
                )
                hsv_distribution_image = HSVColorDistribution(
                    name=values[0],
                    hue_mean=float(values[17]), hue_std=float(values[20]),
                    saturation_mean=float(values[18]), saturation_std=float(values[21]),
                    value_mean=float(values[19]), value_std=float(values[22])
                )
                lab_distribution_image = LABColorDistribution(
                    name=values[0],
                    l_mean=float(values[23]), l_std=float(values[26]),
                    a_mean=float(values[24]), a_std=float(values[27]),
                    b_mean=float(values[25]), b_std=float(values[28])
                )
                rgb_distribution_quantized = RGBColorDistribution(
                    name=values[0],
                    red_mean=float(values[29]), red_std=float(values[32]),
                    green_mean=float(values[30]), green_std=float(values[33]),
                    blue_mean=float(values[31]), blue_std=float(values[34])
                )
                hsv_distribution_quantized = HSVColorDistribution(
                    name=values[0],
                    hue_mean=float(values[35]), hue_std=float(values[38]),
                    saturation_mean=float(values[36]), saturation_std=float(values[39]),
                    value_mean=float(values[37]), value_std=float(values[40])
                )
                lab_distribution_quantized = LABColorDistribution(
                    name=values[0],
                    l_mean=float(values[41]), l_std=float(values[44]),
                    a_mean=float(values[42]), a_std=float(values[45]),
                    b_mean=float(values[43]), b_std=float(values[46])
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
                    "lab_distribution_image": lab_distribution_image,
                    "rgb_distribution_quantized": rgb_distribution_quantized,
                    "hsv_distribution_quantized": hsv_distribution_quantized,
                    "lab_distribution_quantized": lab_distribution_quantized
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

# rgb_centroid_distance
def rgb_centroid_distance(rgb_centroid, rgb_distribution):
    """
    Calculate the perceptually weighted Redmean distance between an RGB centroid
    and an RGB distribution, incorporating the distribution's standard deviation.

    Parameters:
        rgb_centroid (tuple): A 3-tuple representing the RGB values of the centroid 
                              (in OpenCV format, 0-255 for each channel).
        rgb_distribution (RGBColorDistribution): An RGBColorDistribution object
                              containing the mean and standard deviation of the 
                              distribution's red, green, and blue channels.

    Returns:
        float: The calculated Redmean distance between the RGB centroid and the
               RGB distribution, adjusted for the distribution's variability.
    """
    # Extract the mean and std deviation of the distribution
    red_mean, red_std = rgb_distribution.red_mean, rgb_distribution.red_std
    green_mean, green_std = rgb_distribution.green_mean, rgb_distribution.green_std
    blue_mean, blue_std = rgb_distribution.blue_mean, rgb_distribution.blue_std

    # Calculate perceptually weighted deltas
    mean_red = (rgb_centroid[0] + red_mean) / 2
    delta_r = (red_mean - rgb_centroid[0]) / (red_std if red_std != 0 else 1)
    delta_g = (green_mean - rgb_centroid[1]) / (green_std if green_std != 0 else 1)
    delta_b = (blue_mean - rgb_centroid[2]) / (blue_std if blue_std != 0 else 1)

    # Incorporate std deviations into Redmean distance
    return sqrt(
        ((2 + mean_red / 256) * delta_r ** 2) +
        (4 * delta_g ** 2) +
        ((2 + (255 - mean_red) / 256) * delta_b ** 2)
    )

# rgb_distribution_distance
def rgb_distribution_distance(dist1, dist2):
    """
    Calculates the Wasserstein-like distance between two RGB distributions,
    incorporating both mean and standard deviation for each color channel.

    The comparison accounts for the spread (standard deviation) of each distribution
    to better reflect the overlap or divergence between distributions.

    Parameters:
        dist1 (RGBColorDistribution): The first RGB color distribution.
        dist2 (RGBColorDistribution): The second RGB color distribution.

    Returns:
        float: The weighted distance between the two RGB distributions.
    """
    # Define a helper function to incorporate both mean and std
    def channel_distance(mean1, std1, mean2, std2):
        # Weighted difference based on standard deviation
        return abs(mean1 - mean2) / (std1 + std2 + 1e-8)  # Add epsilon to avoid division by zero

    # Calculate distances for each channel
    red_distance = channel_distance(dist1.red_mean, dist1.red_std, dist2.red_mean, dist2.red_std)
    green_distance = channel_distance(dist1.green_mean, dist1.green_std, dist2.green_mean, dist2.green_std)
    blue_distance = channel_distance(dist1.blue_mean, dist1.blue_std, dist2.blue_mean, dist2.blue_std)

    # Combine distances (weighted equally across channels)
    return (red_distance + green_distance + blue_distance) / 3

# hsv_centroid_distance
def hsv_centroid_distance(hsv_centroid, hsv_distribution):
    """
    Calculates the 3D circular distance between an HSV centroid and an HSV distribution.
    
    This metric combines a circular difference for hue and a weighted Euclidean distance
    for saturation (S) and value (V). The standard deviation for each channel is 
    incorporated to account for the distribution's spread.

    OpenCV HSV value ranges:
        - Hue (H): 0-179 (circular scale).
        - Saturation (S): 0-255.
        - Value (V): 0-255.

    Parameters:
        hsv_centroid (tuple): A 3-element tuple representing the H, S, V values of the centroid.
        hsv_distribution (HSVColorDistribution): An HSVColorDistribution object.

    Returns:
        float: The weighted 3D distance between the centroid and the distribution.
    """
    # Circular difference for hue
    hue_diff = min(abs(hsv_centroid[0] - hsv_distribution.hue_mean), 
                   179 - abs(hsv_centroid[0] - hsv_distribution.hue_mean)) / 179.0
    
    # Saturation and value differences
    delta_s = (hsv_centroid[1] - hsv_distribution.saturation_mean) / hsv_distribution.saturation_std if hsv_distribution.saturation_std != 0 else hsv_centroid[1] - hsv_distribution.saturation_mean
    delta_v = (hsv_centroid[2] - hsv_distribution.value_mean) / hsv_distribution.value_std if hsv_distribution.value_std != 0 else hsv_centroid[2] - hsv_distribution.value_mean

    # Weighted 3D distance
    return sqrt(hue_diff ** 2 + delta_s ** 2 + delta_v ** 2)

# hsv_distribution_distance
def hsv_distribution_distance(dist1, dist2):
    """
    Calculates the adapted Wasserstein-like distance between two HSV distributions.
    
    This metric considers the circular nature of hue (H) while combining weighted 
    Euclidean differences for saturation (S) and value (V). The standard deviations 
    of the distributions are incorporated to normalize the contributions of each channel.

    OpenCV HSV value ranges:
        - Hue (H): 0-179 (circular scale).
        - Saturation (S): 0-255.
        - Value (V): 0-255.

    Parameters:
        dist1 (HSVColorDistribution): The first HSV distribution.
        dist2 (HSVColorDistribution): The second HSV distribution.

    Returns:
        float: The adapted 3D distance between the two distributions.
    """
    # Circular difference for hue
    hue_diff = min(abs(dist1.hue_mean - dist2.hue_mean), 
                   179 - abs(dist1.hue_mean - dist2.hue_mean)) / 179.0
    
    # Normalized differences for saturation and value
    sat_diff = (dist1.saturation_mean - dist2.saturation_mean) / (
        sqrt(dist1.saturation_std ** 2 + dist2.saturation_std ** 2)
        if dist1.saturation_std and dist2.saturation_std else 1
    )
    val_diff = (dist1.value_mean - dist2.value_mean) / (
        sqrt(dist1.value_std ** 2 + dist2.value_std ** 2)
        if dist1.value_std and dist2.value_std else 1
    )

    # Weighted 3D distance
    return sqrt(hue_diff ** 2 + sat_diff ** 2 + val_diff ** 2)

# lab_centroid_distance
def lab_centroid_distance(lab_centroid, lab_distribution):
    """
    Calculates the weighted Euclidean distance between a LAB centroid and a LAB distribution.
    
    This function converts the OpenCV LAB representation to the standard LAB ranges:
        - L: 0-100
        - A: -128 to 127
        - B: -128 to 127

    The distance metric accounts for the variance (standard deviation) of the distribution 
    in each channel, weighting differences appropriately. If the variance is zero, a 
    default value of 1 is used to avoid division errors.

    Parameters:
        lab_centroid (tuple): LAB centroid in OpenCV format (L: 0-255, A: 0-255, B: 0-255).
        lab_distribution (LABColorDistribution): The LAB distribution to compare against.

    Returns:
        float: The weighted Euclidean distance between the centroid and the distribution.
    """
    # Normalize centroid values
    l_centroid = lab_centroid[0] * (100 / 255)
    a_centroid = lab_centroid[1] - 128
    b_centroid = lab_centroid[2] - 128

    # Normalize distribution values
    l_mean = lab_distribution.l_mean * (100 / 255)
    a_mean = lab_distribution.a_mean - 128
    b_mean = lab_distribution.b_mean - 128

    delta_l = l_centroid - l_mean
    delta_a = a_centroid - a_mean
    delta_b = b_centroid - b_mean

    return sqrt(
        (delta_l ** 2 / (lab_distribution.l_std ** 2 if lab_distribution.l_std != 0 else 1)) +
        (delta_a ** 2 / (lab_distribution.a_std ** 2 if lab_distribution.a_std != 0 else 1)) +
        (delta_b ** 2 / (lab_distribution.b_std ** 2 if lab_distribution.b_std != 0 else 1))
    )

# lab_distribution_distance
def lab_distribution_distance(dist1, dist2, l_weight=0.6, ab_weight=1.0):
    """
    Calculates a Wasserstein-like distance between two LAB distributions.

    This function compares two LAB distributions by normalizing their means 
    to the standard LAB ranges and approximating their overlap using absolute 
    differences in means and standard deviations for each channel. OpenCV LAB 
    values are normalized as follows:
        - L: 0-100
        - A: -128 to 127
        - B: -128 to 127

    Parameters:
        dist1 (LABColorDistribution): The first LAB color distribution.
        dist2 (LABColorDistribution): The second LAB color distribution.
        l_weight (float): The weight of L channel.
        ab_weight (float): The weight of A and B channels.

    Returns:
        float: The Wasserstein-like distance between the two LAB distributions.
    """
    # Normalize means
    l_mean_1 = dist1.l_mean * (100 / 255)
    a_mean_1 = dist1.a_mean - 128
    b_mean_1 = dist1.b_mean - 128

    l_mean_2 = dist2.l_mean * (100 / 255)
    a_mean_2 = dist2.a_mean - 128
    b_mean_2 = dist2.b_mean - 128

    # Approximate Wasserstein distance
    l_dist = abs(l_mean_1 - l_mean_2) + abs(dist1.l_std - dist2.l_std)
    a_dist = abs(a_mean_1 - a_mean_2) + abs(dist1.a_std - dist2.a_std)
    b_dist = abs(b_mean_1 - b_mean_2) + abs(dist1.b_std - dist2.b_std)

    weight_sum = l_weight + (2 * ab_weight)

    return (l_weight * l_dist) + (ab_weight * a_dist) + (ab_weight * b_dist) / weight_sum

def main(input_color_cluster_file_name, output_location, output_results_file_name):
    """
    Main function.

    Parameters:
        input_color_cluster_file_name (string): File containing input cluster color centroid names from previous script.
        output_location (string): Path to store plots and files.
        output_results_file_name (string): File to write output.
    """

    # Load the data
    df = load_cluster_data(input_color_cluster_file_name)

    # Output file
    output_summary_file_name = os.path.join(output_location, output_results_file_name)
    output_summary_file = open(output_summary_file_name, "w")

    # Define the standard DAB-brown distribution in RGB, HSV and LAB spaces
    dab_brown_centroid_rgb = RGBColorDistribution(
        name="DAB Brown",
        red_mean=140, red_std=20,
        green_mean=110, green_std=20,
        blue_mean=90, blue_std=20
    )

    dab_brown_distribution_rgb = RGBColorDistribution(
        name="DAB Brown",
        red_mean=140, red_std=20,
        green_mean=110, green_std=20,
        blue_mean=90, blue_std=20
    )

    dab_brown_centroid_hsv = HSVColorDistribution(
        name="DAB Brown",
        hue_mean=20, hue_std=5,       # Hue is cyclic and centered at 20
        saturation_mean=100, saturation_std=10,  # Saturation around 100
        value_mean=80, value_std=10   # Value around 80
    )

    dab_brown_distribution_hsv = HSVColorDistribution(
        name="DAB Brown",
        hue_mean=20, hue_std=5,       # Hue is cyclic and centered at 20
        saturation_mean=100, saturation_std=10,  # Saturation around 100
        value_mean=80, value_std=10   # Value around 80
    )

    dab_brown_centroid_lab = LABColorDistribution(
        name="DAB Brown",
        l_mean=50, l_std=10,    # Lightness component
        a_mean=10, a_std=5,     # Green-red axis component
        b_mean=20, b_std=5      # Blue-yellow axis component
    )

    dab_brown_distribution_lab = LABColorDistribution(
        name="DAB Brown",
        l_mean=50, l_std=10,    # Lightness component
        a_mean=10, a_std=5,     # Green-red axis component
        b_mean=20, b_std=5      # Blue-yellow axis component
    )

    # Iterate through images and colors
    for image_name, row in iterate_images_and_colors(df):

        # Centroids
        centroid_rgb = (row["red_centroid"], row["green_centroid"], row["blue_centroid"])
        centroid_hsv = (row["hue_centroid"], row["saturation_centroid"], row["value_centroid"])
        centroid_lab = (row["L_centroid"], row["A_centroid"], row["B_centroid"])

        # Distributions
        rgb_distribution_image = row["rgb_distribution_image"]
        hsv_distribution_image = row["hsv_distribution_image"]
        lab_distribution_image = row["lab_distribution_image"]
        rgb_distribution_quantized = row["rgb_distribution_quantized"]
        hsv_distribution_quantized = row["hsv_distribution_quantized"]
        lab_distribution_quantized = row["lab_distribution_quantized"]

        # Calculate centroid distances
        centroid_distance_rgb = rgb_centroid_distance(centroid_rgb, dab_brown_centroid_rgb)
        centroid_distance_hsv = hsv_centroid_distance(centroid_hsv, dab_brown_centroid_hsv)
        centroid_distance_lab = lab_centroid_distance(centroid_lab, dab_brown_centroid_lab)

        # Calculate Mahalanobis distances for distributions
        distribution_distance_rgb_image = rgb_distribution_distance(rgb_distribution_image, dab_brown_distribution_rgb)
        distribution_distance_rgb_quantized = rgb_distribution_distance(rgb_distribution_quantized, dab_brown_distribution_rgb)
        distribution_distance_hsv_image = hsv_distribution_distance(hsv_distribution_image, dab_brown_distribution_hsv)
        distribution_distance_hsv_quantized = hsv_distribution_distance(hsv_distribution_quantized, dab_brown_distribution_hsv)
        distribution_distance_lab_image = lab_distribution_distance(lab_distribution_image, dab_brown_distribution_lab)
        distribution_distance_lab_quantized = lab_distribution_distance(lab_distribution_quantized, dab_brown_distribution_lab)

        # Write results to output file
        output_summary_file.write(f"# Image\t{row['image_name']}\tColor\t{row['color_name']}\tTotal Pixels\t{row['total_pixels']}:\n")
        output_summary_file.write(f"{row['color_name']}\t")
        output_summary_file.write(f"{centroid_distance_rgb:.2f}\t")
        output_summary_file.write(f"{distribution_distance_rgb_image:.2f}\t")
        output_summary_file.write(f"{distribution_distance_rgb_quantized:.2f}\t")
        output_summary_file.write(f"{centroid_distance_hsv:.2f}\t")
        output_summary_file.write(f"{distribution_distance_hsv_image:.2f}\t")
        output_summary_file.write(f"{distribution_distance_hsv_quantized:.2f}\t")
        output_summary_file.write(f"{centroid_distance_lab:.2f}\t")
        output_summary_file.write(f"{distribution_distance_lab_image:.2f}\t")
        output_summary_file.write(f"{distribution_distance_lab_quantized:.2f}\n")

    # Closing file
    output_summary_file.close()

if __name__ == "__main__":

    # Main Script
    input_color_cluster_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/hypertraining8/results.txt"
    output_location = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/supercolider5/"
    output_results_file_name = "results.txt"

    # Main function
    main(input_color_cluster_file_name, output_location, output_results_file_name)

