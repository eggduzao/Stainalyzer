import openslide
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = 5_000_000_000

def ndpi_to_tiff_with_srgb(ndpi_file, output_folder, downscale_factor=4):
    """
    Convert a large NDPI file to a downscaled TIFF file with an sRGB color profile.

    Parameters:
        ndpi_file (str): Path to the NDPI file.
        output_folder (str): Folder where the TIFF file will be saved.
        downscale_factor (int): Factor by which to downscale the image. Default is 4.
    
    Returns:
        str: Path to the saved TIFF file.
    """
    try:
        # Open the NDPI file with OpenSlide
        slide = openslide.OpenSlide(ndpi_file)
        
        # Get the dimensions of the base layer (highest resolution)
        width, height = slide.dimensions
        
        # Downscaled dimensions
        down_width = width // downscale_factor
        down_height = height // downscale_factor

        # Read the downscaled region
        slide_image = slide.get_thumbnail((down_width, down_height))
        
        # Remove the alpha channel (convert RGBA to RGB)
        slide_image = slide_image.convert("RGB")
        
        # Apply the sRGB color profile
        srgb_profile = ImageCms.createProfile("sRGB")
        slide_image = ImageCms.profileToProfile(slide_image, srgb_profile, srgb_profile, outputMode="RGB")
        
        # Create the output filename
        base_name = os.path.basename(ndpi_file)
        output_file = os.path.join(output_folder, os.path.splitext(base_name)[0] + "_downscaled_srgb.tiff")
        
        # Save as TIFF
        slide_image.save(output_file, format="TIFF")
        print(f"Converted {ndpi_file} to {output_file} with sRGB profile")
        return output_file

    except Exception as e:
        print(f"Error processing {ndpi_file}: {e}")
        return None

# Example usage
if __name__ == "__main__":

    input_location = "/Users/egg/Projects/Stainalyzer/data/DAB_Training/"
    input_ndpi = os.path.join(input_location, "Training1.png")
    #input_ndpi = os.path.join(input_location, "Training2.png")
    #input_ndpi = os.path.join(input_location, "Training3.png")
    #input_ndpi = os.path.join(input_location, "Training4.png")
    output_dir = input_location

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert the NDPI file to TIFF
    ndpi_to_tiff(input_ndpi, output_dir)
    