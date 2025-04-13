
"""
Visualizations
"""

############################################################################################################
### Import
############################################################################################################

import os
import cv2
import torch
import subprocess
import numpy as np
from PIL import Image
from realesrgan.models.realesrgan_model import RealESRGANModel
from basicsr.utils.download_util import load_file_from_url

############################################################################################################
### Constants
############################################################################################################

# Constants
SEED = 1987
np.random.seed(SEED)

############################################################################################################
### Classes
############################################################################################################

class RealESRGAN_Tiler:
    """
    RealESRGAN_Tiler
    """

    def __init__(self, model_path=None, scale=4, tile_size=512, overlap=10, device="cuda", num_workers=5):
        """
        Initializes the Real-ESRGAN tiler.

        Parameters:
            model_path (str): Path to the Real-ESRGAN model weights (.pth file).
            scale (int): Upscaling factor (default=4).
            tile_size (int): Size of each tile before upscaling.
            overlap (int): Overlap between tiles to reduce seams.
            device (str): 'cuda' or 'cpu' for processing.
        """
        self.scale = scale
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device if torch.cuda.is_available() else "cpu"

        # Set the number of threads for PyTorch
        torch.set_num_threads(num_workers)

        # Default model path if not provided
        if model_path is None:
            model_path = os.path.expanduser("/Users/egg/.realesrgan/weights/RealESRGAN_x4plus.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

        # Load the model
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Loads the Real-ESRGAN model.

        Parameters:
            model_path (str): Path to the .pth model file.

        Returns:
            model: Loaded Real-ESRGAN model.
        """

        print(model_path)

        # Explicitly check for GPU availability
        num_gpu = 1 if self.device == "cuda" and torch.cuda.is_available() else 0

        # Ensure the options dictionary has all required fields
        opt = {
            "scale": self.scale,
            "num_gpu": num_gpu,
            "is_train": False,  # Required for inference
            "dist": False,  # Not using distributed training
            "network_g": {  # Define the network architecture
                "type": "RRDBNet",
                "num_in_ch": 3,
                "num_out_ch": 3,
                "num_feat": 64,
                "num_block": 23,
                "num_grow_ch": 32,
                "scale": self.scale
            },
            "path": {
                "param_key_g": "params_ema",
                "pretrain_network_g": model_path,
                "strict_load_g": True
            }
        }

        # Initialize the model
        model = RealESRGANModel(opt=opt)

        # Load the model weights
        state_dict = torch.load(model_path, map_location=self.device)

        # Use "params_ema" if available; otherwise, fallback to "params"
        key = "params_ema" if "params_ema" in state_dict else "params"
        
        if key not in state_dict:
            raise KeyError(f"Unexpected model file format! Available keys: {state_dict.keys()}")

        # Load the correct weights
        model.load_network(state_dict[key], "network_g")

        # Move model to the correct device
        model.model.to(self.device)
        model.model.eval()
        return model

    def process_tile(self, tile):
        """Process a single tile with Real-ESRGAN."""
        img_pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            sr_image = self.model.model(img_pil)
        return cv2.cvtColor(np.array(sr_image), cv2.COLOR_RGB2BGR)

    def tile_image(self, img):
        """
        Process an image by tiling, upscaling each tile, and merging them.

        Parameters:
            img (numpy array): Input image.

        Returns:
            upscaled_image (numpy array): The full upscaled image.
        """
        h, w, _ = img.shape
        new_h, new_w = h * self.scale, w * self.scale
        upscaled_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        for y in range(0, h, self.tile_size - self.overlap):
            for x in range(0, w, self.tile_size - self.overlap):
                tile = img[y:y + self.tile_size, x:x + self.tile_size]
                upscaled_tile = self.process_tile(tile)

                # Define paste coordinates in upscaled image
                y_out, x_out = y * self.scale, x * self.scale
                h_out, w_out = upscaled_tile.shape[:2]
                upscaled_image[y_out:y_out + h_out, x_out:x_out + w_out] = upscaled_tile

        return upscaled_image

    def upscale_image(self, image_path, output_path):
        """
        Upscale an image using tiling.

        Parameters:
            image_path (str): Path to the input image.
            output_path (str): Path to save the upscaled image.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read input image.")

        upscaled_image = self.tile_image(img)
        cv2.imwrite(output_path, upscaled_image)
        print(f"Upscaled image saved to {output_path}")


if __name__ == "__main__":

    # Path to program
    program_location = "/Users/egg/Desktop/realesrgan-ncnn-vulkan-v0.2.0-macos/realesrgan-ncnn-vulkan"

    # Path to the models downloaded
    model_path = "/Users/egg/.realesrgan/weights/"

    # Parameters
    scale = 5
    tile_size = 62


    # Input and Output Locations
    prefix = "/Users/egg/Projects/Stainalyzer/data/"
    input_path = os.path.join(prefix, f"interim/images_to_enhance/cell_00058/0.0.png")
    output_path = os.path.join(prefix, f"interim/images_to_enhance/cell_00058/0.0_{scale}_{tile_size}.png")

    command =  [program_location, 
                "-i", input_path, 
                "-o", output_path, 
                "-n", "realesrgan-x4plus",
                "-s", f"{scale}",
                "-g", "0",
                "-t", f"{tile_size}",
                #"-m", model_path,
                "-f", "png"
                "-v"]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Esrgan test parameters
    # tile_size = 62
    # scale = 10
    # overlap = 22

    # # Print available keys
    # keys = state_dict.keys()
    # for k in keys:
    #     print(f"{k}")
    #     for k2, value in state_dict[k].items():
    #         print(f"{k2} = {type(value)}")

    # # Initialize the Real-ESRGAN tiler
    # tiler = RealESRGAN_Tiler(model_path=model_path,
    #                          scale=scale,
    #                          tile_size=tile_size, 
    #                          overlap=overlap, 
    #                          device="cpu", 
    #                          num_workers=4)

    # # Upscale an image
    # tiler.upscale_image(input_path, output_path)


