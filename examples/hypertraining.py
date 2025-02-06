import os
import cv2
import numpy as np
from shutil import move

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

# HSV distributions for common colors

BLUE_PARAMS = {
    "hue": (120, 10),  # Mean = 120, Std = 10 (Covers cyan to blue)
    "saturation": (100, 50),  # Dull blue
    "value": (150, 50)  # Medium brightness
}

BROWN_PARAMS = {
    "hue": (20, 20),  # Warm brown
    "saturation": (150, 50),  # Medium saturation
    "value": (100, 50)  # Darker tones of brown
}

DEEPBLUE_PARAMS = {
    "hue": (120, 15),  # Slightly broader range for deep blue
    "saturation": (220, 30),  # High saturation for deep tones
    "value": (100, 40)  # Low brightness for "deep" effect
}

GREEN_PARAMS = {
    "hue": (60, 15),  # Covers lime to pure green
    "saturation": (200, 50),  # Vibrant green
    "value": (150, 50)  # Medium brightness
}

ORANGE_PARAMS = {
    "hue": (15, 10),  # Warm orange tones
    "saturation": (220, 40),  # Vibrant orange
    "value": (200, 50)  # Bright orange
}

PURPLE_PARAMS = {
    "hue": (150, 10),  # Lavender to violet shades
    "saturation": (200, 50),  # Vibrant purple
    "value": (150, 50)  # Medium brightness
}

RED_PARAMS = {
    "hue": (0, 10),  # Red tones
    "saturation": (220, 40),  # Vibrant red
    "value": (150, 50)  # Medium brightness
}

YELLOW_PARAMS = {
    "hue": (30, 10),  # Yellow tones
    "saturation": (220, 40),  # Vibrant yellow
    "value": (200, 50)  # Bright yellow
}

GRAY_PARAMS = {
    "hue": (0, 179),  # Hue is irrelevant for gray
    "saturation": (0, 50),  # Low saturation
    "value": (100, 30)  # Medium brightness for gray
}

WHITE_PARAMS = {
    "hue": (0, 179),  # Hue is irrelevant for white
    "saturation": (0, 30),  # Almost no saturation
    "value": (230, 25)  # Very high brightness
}

# Define color distributions (example distributions from earlier)
COLOR_DISTRIBUTIONS = {
    "BLUE": {"hue": (120, 10), "saturation": (200, 50), "value": (150, 50)},
    "BROWN": {"hue": (20, 10), "saturation": (150, 50), "value": (100, 50)},
    "DEEPBLUE": {"hue": (120, 15), "saturation": (220, 30), "value": (100, 40)},
    "GRAY": {"hue": (0, 179), "saturation": (0, 50), "value": (100, 30)},
    "GREEN": {"hue": (60, 15), "saturation": (200, 50), "value": (150, 50)},
    "ORANGE": {"hue": (15, 10), "saturation": (220, 40), "value": (200, 50)},
    "PURPLE": {"hue": (150, 10), "saturation": (200, 50), "value": (150, 50)},
    "RED": {"hue": (0, 10), "saturation": (220, 40), "value": (150, 50)},
    "WHITE": {"hue": (0, 179), "saturation": (0, 30), "value": (230, 25)},
    "YELLOW": {"hue": (30, 10), "saturation": (220, 40), "value": (200, 50)},
}

# Gaussian probability function
def gaussian_probability(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

# Function to calculate posterior probabilities for a pixel
def calculate_posterior_probabilities(h, s, v):
    probabilities = {}
    for color, params in COLOR_DISTRIBUTIONS.items():
        # Calculate likelihoods for each channel
        likelihood_hue = gaussian_probability(h, params["hue"][0], params["hue"][1])
        likelihood_saturation = gaussian_probability(s, params["saturation"][0], params["saturation"][1])
        likelihood_value = gaussian_probability(v, params["value"][0], params["value"][1])

        # Combine likelihoods (assuming independence)
        probabilities[color] = likelihood_hue * likelihood_saturation * likelihood_value

    return probabilities

# Iterate through kernels
input_folder = "/Users/egg/Desktop/Neila/kernels_with_posteriors/"  # Change to the folder containing kernel images
output_folder = "/Users/egg/Desktop/Neila/kernels_with_posteriors/"  # Change to your desired output folder
os.makedirs(output_folder, exist_ok=True)

# Create folders for each color
for color in COLOR_DISTRIBUTIONS.keys():
    os.makedirs(os.path.join(output_folder, color), exist_ok=True)

# Iterate through kernel images
for file_name in os.listdir(input_folder):
    if file_name.startswith("kernel_") and file_name.endswith(".png"):
        file_path = os.path.join(input_folder, file_name)

        # Load the image
        image = cv2.imread(file_path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calculate posterior probabilities for the image
        height, width, _ = hsv_image.shape
        color_scores = {color: 0 for color in COLOR_DISTRIBUTIONS.keys()}

        for i in range(height):
            for j in range(width):
                h, s, v = hsv_image[i, j]
                probabilities = calculate_posterior_probabilities(h, s, v)

                # Add to the cumulative score for each color
                for color, prob in probabilities.items():
                    color_scores[color] += prob

        # Determine the dominant color
        dominant_color = max(color_scores, key=color_scores.get)

        # Move the image to the appropriate folder
        target_folder = os.path.join(output_folder, dominant_color)
        move(file_path, os.path.join(target_folder, file_name))

########


"""
def list_images(directory):
    "" "
    Returns a list of absolute paths to all image files in the given directory.

    :param directory: The path to the directory.
    :return: List of absolute file paths.
    "" "
    # Filter files with common image extensions
    image_extensions = {".png"}
    return [
        os.path.abspath(os.path.join(directory, file))
        for file in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file)) and os.path.splitext(file)[1].lower() in image_extensions
    ]

def calculate_hsv_statistics(image_paths):
    "" "
    Calculate the mean and standard deviation of HSV channels for a list of images.

    :param image_paths: List of image file paths.
    :return: Dictionary with mean and std for Hue, Saturation, and Value channels.
    "" "
    hue_values = []
    saturation_values = []
    value_values = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        # Flatten the channels and collect values
        hue_values.extend(h.flatten())
        saturation_values.extend(s.flatten())
        value_values.extend(v.flatten())

    # Calculate mean and standard deviation for each channel
    statistics = {
        "hue": (np.mean(hue_values), np.std(hue_values)),
        "saturation": (np.mean(saturation_values), np.std(saturation_values)),
        "value": (np.mean(value_values), np.std(value_values)),
    }

    return statistics

# Define your directories
blue_path = "./kernels_with_posteriors/BLUE"
brown_path = "./kernels_with_posteriors/BROWN"
deepblue_path = "./kernels_with_posteriors/DEEPBLUE"
gray_path = "./kernels_with_posteriors/GRAY"
green_path = "./kernels_with_posteriors/GREEN"
orange_path = "./kernels_with_posteriors/ORANGE"
purple_path = "./kernels_with_posteriors/PURPLE"
red_path = "./kernels_with_posteriors/RED"
white_path = "./kernels_with_posteriors/WHITE"
yellow_path = "./kernels_with_posteriors/YELLOW"

# Generate lists of absolute paths
blue_images = list_images(blue_path)
brown_images = list_images(brown_path)
deepblue_images = list_images(deepblue_path)
gray_images = list_images(gray_path)
green_images = list_images(green_path)
orange_images = list_images(orange_path)
purple_images = list_images(purple_path)
red_images = list_images(red_path)
white_params = list_images(white_path)
yellow_images = list_images(yellow_path)

# Calculate statistics for both categories
blue_params = calculate_hsv_statistics(blue_images)
brown_params = calculate_hsv_statistics(brown_images)
deepblue_params = calculate_hsv_statistics(deepblue_images)
gray_params = calculate_hsv_statistics(gray_images)
green_params = calculate_hsv_statistics(green_images)
orange_params = calculate_hsv_statistics(orange_images)
purple_params = calculate_hsv_statistics(purple_images)
red_params = calculate_hsv_statistics(red_images)
white_params = calculate_hsv_statistics(white_images)
yellow_params = calculate_hsv_statistics(yellow_images)

# Print the new parameters
print("New BLUE_PARAMS:")
print(blue_params)

print("\nNew BROWN_PARAMS:")
print(brown_params)

print("\nNew DEEPBLUE_PARAMS:")
print(deepblue_params)

print("\nNew GREEN_PARAMS:")
print(green_params)

print("\nNew GRAY_PARAMS:")
print(gray_params)

print("\nNew ORANGE_PARAMS:")
print(orange_params)

print("\nNew PURPLE_PARAMS:")
print(purple_params)

print("\nNew RED_PARAMS:")
print(red_params)

print("\nNew WHITE_PARAMS:")
print(white_params)

print("\nNew YELLOW_PARAMS:")
print(yellow_params)

"""
