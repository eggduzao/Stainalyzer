
"""
Visualizations
"""

############################################################################################################
### Import
############################################################################################################

import gc
import os
import cv2
import psutil
import numpy as np
from skimage import color
from scipy.interpolate import Rbf
from scipy.interpolate import griddata

############################################################################################################
### Constants
############################################################################################################

# Constants
SEED = 1987
np.random.seed(SEED)

############################################################################################################
### Classes
############################################################################################################

class Hyperenhancer:
    """
    Visualizations TODO DESCRIPTION
    """

    def __init__(self, input_path=None, output_path=None):
        """
        Initialize the PlottingUtils class.

        Parameters:
            srgb_profile (object, optional): A color profile for image processing. 
                                             Defaults to an sRGB profile if not provided.
        """

        # Parameters
        self.input_path = input_path if input_path else None
        self.output_path = output_path if output_path else None
        self.image = self._load_image() if self.input_path else None

    def _load_image(self):
        return cv2.imread(self.input_path)

    def upscale_image_griddata(self, scale_factor=5, method='linear'):
        """
        Upscales an image using `griddata` interpolation in the LAB color space.

        Parameters:
        - image: Input image in BGR format.
        - scale_factor: Factor by which the image is upscaled.
        - method: Interpolation method ('linear', 'nearest', 'cubic').

        Returns:
        - upscaled_image: The upscaled image.
        """
        # Convert to LAB color space for perceptual accuracy
        lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        h, w, _ = lab_image.shape

        # Generate original pixel positions
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        points = np.column_stack((x.ravel(), y.ravel()))  # (w*h, 2)

        # Create new upscaled grid
        new_h, new_w = h * scale_factor, w * scale_factor
        new_y, new_x = np.meshgrid(np.linspace(0, h-1, new_h), np.linspace(0, w-1, new_w), indexing='ij')
        new_points = np.column_stack((new_x.ravel(), new_y.ravel()))  # (new_w*new_h, 2)

        # Interpolate per channel
        upscaled_lab = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        for channel in range(3):
            values = lab_image[:, :, channel].ravel()
            new_values = griddata(points, values, new_points, method=method)
            upscaled_lab[:, :, channel] = np.clip(new_values.reshape(new_h, new_w), 0, 255).astype(np.uint8)

        # Convert back to BGR
        upscaled_bgr = cv2.cvtColor(upscaled_lab, cv2.COLOR_LAB2BGR)
        return upscaled_bgr

    def upscale_image_rbf(self, tile=None, scale_factor=5, radius=10, function='multiquadric', epsilon=1e-6):
        """
        Upscales an image using Radial Basis Function (RBF) interpolation in the LAB color space.
        
        Parameters:
        - tile: Input tile (H, W, 3) in BGR format.
        - scale_factor: Factor by which the image is upscaled.
        - radius: Maximum influence distance for interpolation.
        - function: RBF function to use ('multiquadric', 'inverse', 'gaussian', etc.).
        - epsilon: Small value to avoid division errors.
        
        Returns:
        - upscaled_image: The upscaled image after RBF interpolation.
        """
        
        # Convert image to LAB color space for perceptual accuracy
        if tile is None:
            lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            h, w, _ = lab_image.shape
        else:
            lab_image = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB)
            h, w, _ = lab_image.shape            

        # Create the target upscaled grid
        new_h, new_w = h * scale_factor, w * scale_factor
        
        # Original pixel positions
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        y, x = y.flatten(), x.flatten()
        
        # Scaled pixel positions
        new_y, new_x = np.meshgrid(np.linspace(0, h-1, new_h), np.linspace(0, w-1, new_w), indexing='ij')
        new_y, new_x = new_y.flatten(), new_x.flatten()
        
        # Prepare empty upscaled LAB image/tile
        upscaled_lab = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        
        # Perform RBF interpolation per channel (L, A, B)
        for channel in range(3):
            values = lab_image[:, :, channel].flatten()
            
            # Create RBF interpolator
            rbf = Rbf(x, y, values, function=function, smooth=epsilon)
            
            # Interpolate on the new grid
            new_values = rbf(new_x, new_y)
            
            # Clip values and reshape back to 2D grid
            upscaled_lab[:, :, channel] = np.clip(new_values.reshape(new_h, new_w), 0, 255).astype(np.uint8)
        
        # Convert back to BGR for OpenCV display
        upscaled_bgr = cv2.cvtColor(upscaled_lab, cv2.COLOR_LAB2BGR)

        return upscaled_bgr

    def tile_image(self, tile_x, tile_y):
        """
        Splits an image into tiles of size (tile_x, tile_y).
        
        Parameters:
        - tile_x: Number of tiles along width.
        - tile_y: Number of tiles along height.
        
        Returns:
        - tiles: List of image tiles.
        - tile_shape: Shape of each tile.
        """
        h, w, _ = self.image.shape
        tile_h, tile_w = h // tile_y, w // tile_x
        tiles = []
        for i in range(tile_y):
            for j in range(tile_x):
                tile = self.image[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
                tiles.append(tile)
        return tiles, (tile_h, tile_w)

    def merge_tiles(self, tiles, tile_shape, image_shape, tile_x, tile_y):
        """
        Merges tiles back into the original image shape.
        
        Parameters:
        - tiles: List of image tiles.
        - tile_shape: Shape of each tile.
        - image_shape: Shape of the original image.
        - tile_x: Number of tiles along width.
        - tile_y: Number of tiles along height.
        
        Returns:
        - merged_image: Reconstructed image.
        """
        h, w, _ = image_shape
        merged_image = np.zeros((h, w, 3), dtype=np.uint8)
        tile_h, tile_w, _ = tile_shape
        index = 0
        for i in range(tile_y):
            for j in range(tile_x):
                merged_image[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w] = tiles[index]
                index += 1
        return merged_image

    def run_tiled(self, tiles=(4, 4), scale_factor=5, radius=10):

        # Define the number of tiles
        tile_x, tile_y = tiles[0], tiles[1]  # Split into a 4x4 grid

        # Tile the image
        tiles, tile_shape = self.tile_image(tile_x, tile_y)

        # Process each tile (For example, convert to grayscale)
        processed_tiles = []
        for tile in tiles:
            upscaled_tile = self.upscale_image_rbf(tile, scale_factor=scale_factor, radius=radius)
            processed_tiles.append(upscaled_tile)
            del upscaled_tile
            gc.collect()  # Force memory cleanup

        # Define new tile shape after upscaling
        new_tile_shape = (tile_shape[0] * scale_factor, tile_shape[1] * scale_factor, 3)

        # Merge tiles back into a larger upscaled image
        new_image_shape = (self.image.shape[0] * scale_factor, self.image.shape[1] * scale_factor, 3)
        merged_image = self.merge_tiles(processed_tiles, new_tile_shape, new_image_shape, tile_x, tile_y)

        # Save the output
        file_name = os.path.basename(self.input_path)[:-4]
        output_file_name = os.path.join(self.output_path, file_name+"E.png")
        cv2.imwrite(output_file_name, merged_image)

    def run_create_tiles(self, tiles=(4, 4)):

        # Define the number of tiles
        tile_x, tile_y = tiles[0], tiles[1]  # Split into a grid

        # Tile the image
        tiles, tile_shape = self.tile_image(tile_x, tile_y)

        # Process each tile (For example, convert to grayscale)
        counter_x, counter_y = 0, 0
        processed_tiles = []
        for tile in tiles:
            if(counter_y != 0 and counter_y%tile_y == 0):
                counter_x = counter_x + 1
                counter_y = 0
            output_tile_name = os.path.join(self.output_path, f"{counter_x}_{counter_y}.png")
            cv2.imwrite(output_tile_name, tile)
            counter_y = counter_y + 1

if __name__ == "__main__":

    prefix = "/Users/egg/Projects/Stainalyzer/data/"
    input_path = os.path.join(prefix, "interim/images_to_enhance/cell_00058/")
    output_path = os.path.join(prefix, "interim/images_to_enhance/cell_00058/")
    #os.mkdir(output_path)
    #hyper = Hyperenhancer(input_path, output_path)
    #hyper.run_create_tiles(tiles=(10, 10))

    for x in range(0, 10):
        for y in range(0, 10):
            in_path = os.path.join(input_path, f"{x}_{y}.png")
            hyper = Hyperenhancer(in_path, output_path)
            hyper.run_tiled(tiles=(2, 2), scale_factor=10, radius=10)
            break
        break

    #hyper.run_tiled(tiles=(25, 25), scale_factor=5, radius=10)
    #new_image = hyper.upscale_image_rbf(scale_factor=2, radius=10, function='inverse', epsilon=1e-6)
    #print(f"Memory usage before: {psutil.Process(os.getpid()).memory_info().rss / 1e6} MB")

#    # Function (e.g., ‘multiquadric’, ‘inverse’, ‘gaussian’, ‘cubic’)
# 🔹 Different Radial Basis Functions (RBF) influence how smooth or sharp transitions appear.
# 🔹 For sharper results: Try 'gaussian' or 'thin_plate'.
# 🔹 For smoother transitions: 'multiquadric' is often useful.
#    # Radius (Influence Range of Each Pixel in Interpolation)
# 🔹 Lower radius → Pixels are influenced only by closer neighbors → sharper but may introduce artifacts.
# 🔹 Higher radius → Pixels are influenced by farther neighbors → smoother but may blur details.
# 🔹 Best strategy: Test different radii for biological structures.
#    # Epsilon (Smoothing Factor in RBF Interpolation)
# 🔹 Lower epsilon → Sharper, higher contrast, but may overfit and amplify noise.
# 🔹 Higher epsilon → Smoother, denoised, but may lose fine details.
#    # Scale_factor (Upsampling Ratio)
# 🔹 Higher scale_factor (e.g., 10x) → More pixels, but needs stronger interpolation control.
# 🔹 Increasing scale_factor alone won’t add details unless combined with enhanced interpolation.

