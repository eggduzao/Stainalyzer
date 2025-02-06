
# Try to classify the color based on posterior probability and pre-defined colors

# Import packages
import os
import cv2
import numpy as np
from shutil import move
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Initial priors are equally probable as "we don't know" the image yet
PRIOR_BLUE = 0.1
PRIOR_BROWN = 0.1
PRIOR_DEEPBLUE = 0.1
PRIOR_GREEN = 0.1
PRIOR_GRAY = 0.1
PRIOR_ORANGE = 0.1
PRIOR_PURPLE = 0.1
PRIOR_RED = 0.1
PRIOR_WHITE = 0.1
PRIOR_YELLOW = 0.1

# HSV Initial Distributions - BLUE pixels
BLUE_PARAMS = {
    "hue": (120, 10),  # Covers cyan to blue
    "saturation": (100, 50),  # Dull blue
    "value": (150, 50)  # Medium brightness
}

# HSV Initial Distributions - BROWN pixels
BROWN_PARAMS = {
    "hue": (20, 20),  # Warm brown
    "saturation": (150, 50),  # Medium saturation
    "value": (100, 50)  # Darker tones of brown
}

# HSV Initial Distributions - DEEP BLUE pixels
DEEPBLUE_PARAMS = {
    "hue": (120, 15),  # Slightly broader range for deep blue
    "saturation": (220, 30),  # High saturation for deep tones
    "value": (100, 40)  # Low brightness for "deep" effect
}

# HSV Initial Distributions - GREEN pixels
GREEN_PARAMS = {
    "hue": (60, 15),  # Covers lime to pure green
    "saturation": (200, 50),  # Vibrant green
    "value": (150, 50)  # Medium brightness
}

# HSV Initial Distributions - ORANGE pixels
ORANGE_PARAMS = {
    "hue": (15, 10),  # Warm orange tones
    "saturation": (220, 40),  # Vibrant orange
    "value": (200, 50)  # Bright orange
}

# HSV Initial Distributions - PURPLE pixels
PURPLE_PARAMS = {
    "hue": (150, 10),  # Lavender to violet shades
    "saturation": (200, 50),  # Vibrant purple
    "value": (150, 50)  # Medium brightness
}

# HSV Initial Distributions - RED pixels
RED_PARAMS = {
    "hue": (0, 10),  # Red tones
    "saturation": (220, 40),  # Vibrant red
    "value": (150, 50)  # Medium brightness
}

# HSV Initial Distributions - YELLOW pixels
YELLOW_PARAMS = {
    "hue": (30, 10),  # Yellow tones
    "saturation": (220, 40),  # Vibrant yellow
    "value": (200, 50)  # Bright yellow
}

# HSV Initial Distributions - GRAY pixels
GRAY_PARAMS = {
    "hue": (0, 179),  # Hue is irrelevant for gray
    "saturation": (0, 50),  # Low saturation
    "value": (100, 30)  # Medium brightness for gray
}

# HSV Initial Distributions - WHITE pixels
WHITE_PARAMS = {
    "hue": (0, 179),  # Hue is irrelevant for white
    "saturation": (0, 30),  # Almost no saturation
    "value": (230, 25)  # Very high brightness
}

# Define color distributions (example distributions from earlier)
COLOR_DISTRIBUTIONS = {
    "BLUE": BLUE_PARAMS,
    "BROWN": BROWN_PARAMS,
    "DEEPBLUE": DEEPBLUE_PARAMS,
    "GRAY": GRAY_PARAMS,
    "GREEN": GREEN_PARAMS,
    "ORANGE": ORANGE_PARAMS,
    "PURPLE": PURPLE_PARAMS,
    "RED": RED_PARAMS,
    "WHITE": WHITE_PARAMS,
    "YELLOW": YELLOW_PARAMS,
}

# Gaussian probability function
def gaussian_probability(x, mean, std):
    z = (x - mean) / std
    z = np.clip(z, -10, 10)  # Avoid large exponentials
    return np.exp(-0.5 * z**2) / (std * np.sqrt(2 * np.pi))

# Function to calculate posterior probabilities for a pixel
def calculate_posterior_probabilities(h, s, v):

    # Probability vector
    probabilities = []

    # Calculate the posterior probability for each element of an HSV image
    for color, params in COLOR_DISTRIBUTIONS.items():

        # Calculate likelihoods for each channel
        likelihood_hue = gaussian_probability(h, params["hue"][0], params["hue"][1])
        likelihood_saturation = gaussian_probability(s, params["saturation"][0], params["saturation"][1])
        likelihood_value = gaussian_probability(v, params["value"][0], params["value"][1])

        posterior_probability = likelihood_hue * likelihood_saturation * likelihood_value
        probabilities.append(posterior_probability)

    return probabilities

def replace_black_pixels(image):
    """Replace black pixels (RGB = 0, 0, 0) with the nearest non-black pixel."""
    height, width, channels = image.shape
    non_black_coords = np.array(
        [(x, y) for x in range(height) for y in range(width) if not np.all(image[x, y] == [0, 0, 0])]
    )
    non_black_pixels = np.array([image[x, y] for x, y in non_black_coords])
    black_coords = np.array(
        [(x, y) for x in range(height) for y in range(width) if np.all(image[x, y] == [0, 0, 0])]
    )

    if len(non_black_coords) == 0:
        raise ValueError("Image contains only black pixels.")
    
    tree = KDTree(non_black_coords)
    for x, y in black_coords:
        nearest_idx = tree.query([x, y])[1]
        image[x, y] = non_black_pixels[nearest_idx]
    return image

def plot_categories(image_rgb, classification_map, categories, output_folder):

    # Total size of plots
    height, width, _ = image_rgb.shape

    # Iterate over categories
    for category_idx, category_name in enumerate(categories):

        # Calculate stats
        mask = classification_map == category_idx
        category_image = np.ones_like(image_rgb) * 255  # White background
        category_image[mask] = image_rgb[mask]
        
        # Calculate stats
        num_pixels = np.sum(mask)
        percentage = (num_pixels / (height * width)) * 100

        # Verify if there are any pixels
        if num_pixels == 0:
            print(f"Warning: No pixels found for category '{category_name}'")
            continue
        
        # Save category image
        plt.imshow(category_image)
        plt.title(f"{category_name}: {num_pixels} pixels ({percentage:.2f}%)")
        plt.axis("off")
        plt.savefig(f"{output_folder}_{category_name}_category.png")
        plt.close()

def plot_histograms(image_rgb, classification_map, categories, output_folder):

    # Iterate over categories
    for category_idx, category_name in enumerate(categories):

        # Calculate stats
        mask = classification_map == category_idx
        pixels = image_rgb[mask]

        # Verify if there are any pixels
        if pixels.size == 0:
            print(f"Warning: No pixels found for category '{category_name}'")
            continue
        
        # Compute histograms
        plt.figure(figsize=(15, 5))
        
        # RGB Histogram
        plt.subplot(1, 3, 1)
        plt.hist(pixels[:, 0], bins=256, color='red', alpha=0.6, label='Red')
        plt.hist(pixels[:, 1], bins=256, color='green', alpha=0.6, label='Green')
        plt.hist(pixels[:, 2], bins=256, color='blue', alpha=0.6, label='Blue')
        plt.title(f"RGB Histogram - {category_name}")
        plt.legend()
        
        # HSV Histogram
        hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        plt.subplot(1, 3, 2)
        plt.hist(hsv_pixels[:, 0], bins=256, color='orange', alpha=0.6, label='Hue')
        plt.hist(hsv_pixels[:, 1], bins=256, color='purple', alpha=0.6, label='Saturation')
        plt.hist(hsv_pixels[:, 2], bins=256, color='yellow', alpha=0.6, label='Value')
        plt.title(f"HSV Histogram - {category_name}")
        plt.legend()
        
        # CMYK Histogram (Optional, Approximation)
        plt.subplot(1, 3, 3)
        cmyk_pixels = 1 - (pixels / 255.0)
        plt.hist(cmyk_pixels[:, 0], bins=256, color='cyan', alpha=0.6, label='Cyan')
        plt.hist(cmyk_pixels[:, 1], bins=256, color='magenta', alpha=0.6, label='Magenta')
        plt.hist(cmyk_pixels[:, 2], bins=256, color='yellow', alpha=0.6, label='Yellow')
        plt.title(f"CMYK Histogram - {category_name}")
        plt.legend()
        
        plt.savefig(f"{output_folder}_{category_name}_histogram.png")
        plt.close()

# Input Output Files
input_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/"
output_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/6520-21E"
output_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/6520-21E.txt"
output_file_name_2 = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/6520-21E_posterior.txt"
output_file = open(output_file_name, "w")
output_file_2 = open(output_file_name_2, "w")

# Iterate through kernel images
for file_name in os.listdir(input_folder):

    # Join file path
    file_path = os.path.join(input_folder, file_name)

    # If not an image, continue
    if(os.path.splitext(file_path)[-1] not in [".png", ".jpg"]):
        continue

    # Load the image and get its total height and width
    image = cv2.imread(file_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = hsv_image.shape

    # If image is not RGB, then convert it to RGB
    if rgb_image.shape[-1] != 3:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    print(f"RGB Image shape: {rgb_image.shape}")
    print(f"RGB Unique colors in image: {np.unique(rgb_image.reshape(-1, 3), axis=0)}")

    print(f"HSV Image shape: {hsv_image.shape}")
    print(f"HSV Unique colors in image: {np.unique(hsv_image.reshape(-1, 3), axis=0)}")

    # For each pixel:
    # If it is black, assign color of nearest non-black pixel
    rgb_image = replace_black_pixels(rgb_image)

    print(f"RGB Image shape: {rgb_image.shape}")
    print(f"RGB Unique colors in image: {np.unique(rgb_image.reshape(-1, 3), axis=0)}")

    print(f"HSV Image shape: {hsv_image.shape}")
    print(f"HSV Unique colors in image: {np.unique(hsv_image.reshape(-1, 3), axis=0)}")

    # Create classification map to store the classification of every pixel
    classification_map = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.int32)

    # Classify each pixel based on the preexisting distributions
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            pixel_h, pixel_s, pixel_v = hsv_image[i, j]
            posterior_probs = calculate_posterior_probabilities(pixel_h, pixel_s, pixel_v)
            classification_map[i, j] = np.argmax(posterior_probs)
            output_file.write("\t".join([str(e) for e in [i, j, pixel_h, pixel_s, pixel_v]])+"\n")
            output_file_2.write("\t".join([str(e) for e in [i, j]+[-np.log10(p) for p in [posterior_probs]]])+"\n")

    # Plot categorized images with white background
    categories = list(COLOR_DISTRIBUTIONS.keys())
    plot_categories(rgb_image, classification_map, categories, output_folder)

    # Create RGB, CYMK and HSV histograms for each case
    plot_histograms(rgb_image, classification_map, categories, output_folder)

# Closing file    
output_file.close()
output_file_2.close()
