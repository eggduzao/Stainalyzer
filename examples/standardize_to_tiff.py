"""
Standardize images into a TIFF format with an embedded sRGB ICC profile.
It processes all images in the input_folder (supports formats like JPG/JPEG, PNG, TIFF, BMP, WEBP)
and saves the standardized TIFF files in output_folder with the same base name.
"""

import io
import os
from PIL import Image, ImageCms

# Paths for input and output folders
input_folder = "/Users/egg/Projects/Stainalyzer/data/DAB_Training/"
output_folder = "/Users/egg/Projects/Stainalyzer/data/DAB_Training/"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to process and save an image as a standardized TIFF
def process_image_to_tiff(input_path, output_path, srgb_profile):
    # Open the image
    image = Image.open(input_path)
    
    # Convert to RGB mode if not already
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Check for a valid ICC profile
    icc_profile_data = image.info.get("icc_profile", None)
    if icc_profile_data:
        try:
            # Use io.BytesIO to handle the embedded profile
            input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile_data))
            print(f"Converting existing profile to sRGB for: {input_path}")
        except Exception as e:
            # If the profile is invalid, fall back to sRGB
            print(f"Invalid profile, assigning sRGB: {input_path}. Error: {e}")
            input_profile = srgb_profile
    else:
        print(f"Assigning sRGB profile to: {input_path}")
        input_profile = srgb_profile

    # Convert the image to the sRGB color space
    image = ImageCms.profileToProfile(image, input_profile, srgb_profile, outputMode="RGB")

    # Save the image as a TIFF with sRGB ICC profile
    image.save(output_path, format="TIFF")
    print(f"Processed: {output_path}")

# Main script to process all images in the input folder
def standardize_images_to_tiff(input_folder, output_folder):
    # Supported image formats
    supported_formats = (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp")

    # Create the sRGB ICC profile
    srgb_profile = ImageCms.createProfile("sRGB")

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):

        if ".tiff" in filename:
            continue

        input_path = os.path.join(input_folder, filename)

        if os.path.isfile(input_path) and filename.lower().endswith(supported_formats):
            # Define output path with .tiff extension
            output_name = os.path.splitext(filename)[0] + ".tiff"
            output_path = os.path.join(output_folder, output_name)

            # Process the image
            process_image_to_tiff(input_path, output_path, srgb_profile)

# Run the script
if __name__ == "__main__":
    standardize_images_to_tiff(input_folder, output_folder)
