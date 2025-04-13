import os
import cv2
import numpy as np


# HSV distributions for common colors

BLUE_PARAMS = {
    "hue": (120, 15),  # Mean = 120, Std = 15 (cyan to deep blue)
    "saturation": (200, 50),  # Vibrant blue
    "value": (150, 50)  # Medium brightness
}

BROWN_PARAMS = {
    "hue": (20, 10),  # Warm brown
    "saturation": (150, 50),  # Medium saturation
    "value": (150, 50)  # Medium brightness
}

DEEPBLUE_PARAMS = {
    "hue": (240, 10),  # Darker blue (deep blue)
    "saturation": (200, 50),  # Vibrant deep blue
    "value": (100, 30)  # Darker brightness
}

GREEN_PARAMS = {
    "hue": (90, 20),  # Green (lime to forest green)
    "saturation": (200, 50),  # Vibrant green
    "value": (150, 50)  # Medium brightness
}

ORANGE_PARAMS = {
    "hue": (30, 10),  # Orange-yellow tones
    "saturation": (200, 50),  # Vibrant orange
    "value": (200, 50)  # Bright
}

PURPLE_PARAMS = {
    "hue": (270, 15),  # Purple shades
    "saturation": (200, 50),  # Vibrant purple
    "value": (150, 50)  # Medium brightness
}

RED_PARAMS = {
    "hue": (0, 10),  # Red tones
    "saturation": (200, 50),  # Vibrant red
    "value": (150, 50)  # Medium brightness
}

YELLOW_PARAMS = {
    "hue": (60, 10),  # Yellow shades
    "saturation": (200, 50),  # Vibrant yellow
    "value": (100, 50)  # Dull yellow
}


def list_images(directory):
    """
    Returns a list of absolute paths to all image files in the given directory.

    :param directory: The path to the directory.
    :return: List of absolute file paths.
    """
    # Filter files with common image extensions
    image_extensions = {".png"}
    return [
        os.path.abspath(os.path.join(directory, file))
        for file in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file)) and os.path.splitext(file)[1].lower() in image_extensions
    ]

def calculate_hsv_statistics(image_paths):
    """
    Calculate the mean and standard deviation of HSV channels for a list of images.

    :param image_paths: List of image file paths.
    :return: Dictionary with mean and std for Hue, Saturation, and Value channels.
    """
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

print("\nNew ORANGE_PARAMS:")
print(orange_params)

print("\nNew PURPLE_PARAMS:")
print(purple_params)

print("\nNew RED_PARAMS:")
print(red_params)

print("\nNew YELLOW_PARAMS:")
print(yellow_params)


