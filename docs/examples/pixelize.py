
######################################################################################################
# Contagem de Pixels
#   3752-20     913916
#   7716-20G    798591
#   8414-20     0
######################################################################################################

"""
Explanation of the Code

    1.  Dynamic Thresholds:
    •   The calculate_dynamic_thresholds function calculates lower and upper thresholds for each HSV channel based on percentiles. For example:
    •   The 5th percentile of Hue ensures we capture the lower bound of brownish hues.
    •   The 20th percentile captures the upper range of brownish colors.
    •   These thresholds adapt to the color distributions in your specific image.
    2.  Mask Creation:
    •   Pixels falling within the dynamically calculated bounds are marked as foreground, forming the mask.
    3.  Morphological Operations:
    •   The cv2.morphologyEx function applies a closing operation to remove small holes in the mask and smooth the edges.
    4.  Percentage Calculation:
    •   The code calculates the fraction of the image covered by stained regions and outputs the percentage.
    5.  Visualization:
    •   The script shows:
    •   The original image.
    •   The binary mask highlighting stained regions.
    •   An overlay of the mask on the original image.

Customizable Parameters

    1.  Percentiles:
    •   You can adjust the percentiles in calculate_dynamic_thresholds to fine-tune the detection. For example:
    •   Brownish stains: Hue (5-20%), Saturation (30-100%), Value (10-90%).
    2.  Morphological Kernel Size:
    •   The kernel size in cv2.getStructuringElement can be changed to control how aggressively small holes or noise are cleaned.
    3.  Color Ranges:
    •   If the staining protocol uses a different chromogen (e.g., AEC instead of DAB), adjust the thresholds accordingly.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "./8414-20F ACE-2 10X(2).jpg"  # Replace with the path to your image
image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define a function to dynamically calculate thresholds based on histograms
def calculate_dynamic_thresholds(hsv, channel, lower_percentile, upper_percentile):
    """
    Dynamically calculate lower and upper thresholds for a specific HSV channel.
    :param hsv: The HSV image.
    :param channel: The channel index (0: Hue, 1: Saturation, 2: Value).
    :param lower_percentile: The lower percentile for the threshold.
    :param upper_percentile: The upper percentile for the threshold.
    :return: Lower and upper threshold values.
    """
    hist = cv2.calcHist([hsv], [channel], None, [256], [0, 256])
    cdf = np.cumsum(hist) / hist.sum()  # Calculate cumulative distribution function
    
    lower_threshold = np.argmax(cdf >= lower_percentile / 100.0)
    upper_threshold = np.argmax(cdf >= upper_percentile / 100.0)
    
    return lower_threshold, upper_threshold

# Dynamically calculate thresholds for each channel
hue_lower, hue_upper = calculate_dynamic_thresholds(hsv_image, 0, 5, 20)  # Hue thresholds
sat_lower, sat_upper = calculate_dynamic_thresholds(hsv_image, 1, 30, 100)  # Saturation thresholds
val_lower, val_upper = calculate_dynamic_thresholds(hsv_image, 2, 10, 90)  # Value thresholds

# Apply the thresholds to create a mask
lower_bound = np.array([hue_lower, sat_lower, val_lower])
upper_bound = np.array([hue_upper, sat_upper, val_upper])
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Use morphological operations to clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Calculate the percentage of stained pixels
stained_pixels = np.sum(mask_cleaned > 0)
total_pixels = mask_cleaned.size
percentage_stained = (stained_pixels / total_pixels) * 100

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("HSV Mask (Stained Regions)")
plt.imshow(mask_cleaned, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Overlay of Mask on Original")
overlay = cv2.bitwise_and(image, image, mask=mask_cleaned)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

plt.suptitle(f"Percentage of Stained Pixels: {percentage_stained:.2f}%")
plt.suptitle(f"Total of Stained Pixels: {stained_pixels} pxs")
plt.tight_layout()
plt.show()

