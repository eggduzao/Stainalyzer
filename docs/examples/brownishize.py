import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define a function to dynamically calculate thresholds based on histograms
def calculate_dynamic_thresholds(hsv, channel, lower_percentile, upper_percentile):
    hist = cv2.calcHist([hsv], [channel], None, [256], [0, 256])
    cdf = np.cumsum(hist) / hist.sum()  # Calculate cumulative distribution function   
    lower_threshold = np.argmax(cdf >= lower_percentile / 100.0)
    upper_threshold = np.argmax(cdf >= upper_percentile / 100.0)
    return lower_threshold, upper_threshold

# Load the image
image_path = "./7716-20G ACE-2 40X(2).jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Convert to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

########################

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

########################

reconstructed_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_cleaned)

# Split into H, S, and V channels
hue, saturation, value = cv2.split(reconstructed_image)

# Combine the HSV channels into one image
combined_hsv = cv2.merge([hue, saturation, value])

# Convert back to RGB for visualization
combined_rgb = cv2.cvtColor(combined_hsv, cv2.COLOR_HSV2RGB)

# Plot the individual HSV channels
plt.figure(figsize=(15, 4))
plt.subplot(1, 4, 1)
plt.title("Hue Channel")
plt.imshow(hue, cmap='gray')
plt.colorbar()

plt.subplot(1, 4, 2)
plt.title("Saturation Channel")
plt.imshow(saturation, cmap='gray')
plt.colorbar()

plt.subplot(1, 4, 3)
plt.title("Value Channel")
plt.imshow(value, cmap='gray')
plt.colorbar()

plt.subplot(1, 4, 4)
plt.title("Combined")
plt.imshow(combined_rgb)
plt.colorbar()

plt.tight_layout()
plt.show()