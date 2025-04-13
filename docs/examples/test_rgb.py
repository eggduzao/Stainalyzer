"""
Common modes:
    RGB: Red-Green-Blue (color image).
    L: Grayscale (single channel).
    RGBA: RGB with Alpha (transparency channel).
    CMYK: Cyan-Magenta-Yellow-Black (used for printing).

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
"""

import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("./input/3753-21C ACE-2 40X(2).jpg")

# Convert BGR to RGB
image_rgb = image # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot the channels
colors = ('r', 'g', 'b')
for i, color in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    plt.plot(histogram, color=color)
    plt.title("RGB Channel Histograms")
plt.show()

