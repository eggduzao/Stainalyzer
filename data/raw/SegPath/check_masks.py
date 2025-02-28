
import os
import cv2
import numpy as np

def check_black_pixels(directory):
    """
    Iterates over all *_mask.png images in the specified directory and checks if they contain only black pixels (RGB 0,0,0).
    If any non-black pixels are found, their unique values are printed.
    
    Parameters:
    directory (str): Path to the directory containing the mask images.
    """
    for filename in os.listdir(directory):
        if filename.endswith("_mask.png"):  # Process only mask files
            filepath = os.path.join(directory, filename)
            
            # Read image in RGB mode
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Error reading {filename}. Skipping...")
                continue

            # Reshape to a list of RGB values
            reshaped_img = img.reshape(-1, 3)

            # Get unique colors in the image
            unique_colors = np.unique(reshaped_img, axis=0)

            # Convert unique colors to a list of tuples
            unique_colors_list = [tuple(color) for color in unique_colors]

            # Check if all pixels are black
            if len(unique_colors_list) == 1 and unique_colors_list[0] == (0, 0, 0):
                print(f"{filename}\n  All 000")
            else:
                print(f"{filename}\n  Found: {', '.join(map(str, unique_colors_list))}")

if __name__ == "__main__":

	# Example usage: Change "your_directory_path" to the folder containing the mask images
	root_dir = "/Users/egg/Projects/Stainalyzer/data/aSMA/"
	directories = [
		os.path.join(root_dir, "aSMA_SmoothMuscle"),
		os.path.join(root_dir, "CD3CD20_Lymphocyte")
	]
	for directory_path in directories:
		check_black_pixels(directory_path)

