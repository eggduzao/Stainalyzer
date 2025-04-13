import cv2
import numpy as np
import pandas as pd

# Directory paths to the uploaded images
image_paths = [
    "input/3753-21C ACE-2 40X(2).jpg",
    "input/3833-21H HLA-G G2 40X(2).jpg",
    "input/5086-21F ACE-2 40X(4).jpg",
    "input/5829-21B ACE-2 40X(4).jpg",
    "input/6520-21E ACE-2 40X(2).jpg",
    "input/7716-20G ACE-2 40X(2).jpg",
    "input/8414-20F ACE-2 10X(2).jpg",
]

# Define HSV thresholds for DAB staining (brownish regions)
LOWER_BROWN = np.array([10, 50, 50])  # Lower HSV bounds for brown
UPPER_BROWN = np.array([40, 255, 255])  # Upper HSV bounds for brown

results = {}

for idx, image_path in enumerate(image_paths, start=1):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        continue

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the brownish regions
    brown_mask = cv2.inRange(hsv_image, LOWER_BROWN, UPPER_BROWN)

    # Calculate the total number of pixels and the number of brownish pixels
    total_pixels = brown_mask.size
    stained_pixels = np.count_nonzero(brown_mask)

    # Calculate percentage of stained area
    stained_percentage = (stained_pixels / total_pixels) * 100

    # Save results
    results[f"Image_{idx}"] = {
        "Total Pixels": total_pixels,
        "Stained Pixels": stained_pixels,
        "Stained Percentage": stained_percentage,
    }

# Convert results to a DataFrame and print
df_results = pd.DataFrame.from_dict(results, orient="index")
print(df_results)