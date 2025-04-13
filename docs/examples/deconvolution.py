
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_od(image):
    "" "Convert RGB image to Optical Density (OD) space." ""
    image = image.astype(np.float32) / 255  # Normalize to range [0, 1]
    image[image == 0] = 1e-6  # Avoid log(0)
    od = -np.log(image)
    return od

def od_to_rgb(od_image):
    "" "Convert Optical Density (OD) space back to RGB." ""
    image = np.exp(-od_image)
    image = (image * 255).astype(np.uint8)
    return image

def color_deconvolution(image, stain_matrix):
    "" "Perform color deconvolution on the image." ""
    od_image = rgb_to_od(image)
    # Normalize stain matrix
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)
    # Deconvolution
    deconvolved = np.dot(od_image.reshape((-1, 3)), np.linalg.pinv(stain_matrix).T)
    deconvolved = deconvolved.reshape((image.shape[0], image.shape[1], -1))
    return deconvolved

def plot_channels(deconvolved, labels, original_image):
    "" "Plot deconvoluted channels alongside the original image." ""
    fig, axs = plt.subplots(1, len(deconvolved) + 1, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    for i, channel in enumerate(deconvolved):
        axs[i + 1].imshow(channel, cmap='gray')
        axs[i + 1].set_title(labels[i])
        axs[i + 1].axis("off")
    
    plt.tight_layout()
    plt.show()

# Example stain matrix (for DAB and Hematoxylin)
# Adjust these values for your specific staining protocol
stain_matrix = np.array([
    [0.65, 0.70, 0.29],  # DAB
    [0.07, 0.99, 0.11],  # Hematoxylin
    [0.27, 0.57, 0.78]   # Residual/background
])

# Load the input image (replace 'path/to/image.jpg' with your file path)
input_image = cv2.imread('./input/3753-21C ACE-2 40X(2).jpg')

# Perform color deconvolution
deconvolved_channels = color_deconvolution(input_image, stain_matrix)

# Normalize and split channels
dab_channel = deconvolved_channels[:, :, 0]
hematoxylin_channel = deconvolved_channels[:, :, 1]
residual_channel = deconvolved_channels[:, :, 2]

# Threshold for DAB-negative regions
dab_positive = (dab_channel > np.percentile(dab_channel, 50)).astype(np.float32)
dab_negative = 1 - dab_positive

# Plot the results
plot_channels(
    [dab_channel, dab_positive, dab_negative, hematoxylin_channel, residual_channel],
    ["DAB Channel", "DAB Positive", "DAB Negative", "Hematoxylin", "Residual"],
    input_image
)

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_od(image):
    """Convert RGB image to Optical Density (OD) space."""
    image = image.astype(np.float32) / 255  # Normalize to range [0, 1]
    image[image == 0] = 1e-6  # Avoid log(0)
    od = -np.log(image)
    return od

def od_to_rgb(od_image):
    """Convert Optical Density (OD) space back to RGB."""
    image = np.exp(-od_image)
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image

def color_deconvolution(image, stain_matrix):
    """Perform color deconvolution on the image."""
    od_image = rgb_to_od(image)
    # Normalize stain matrix
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)
    # Deconvolution
    deconvolved = np.dot(od_image.reshape((-1, 3)), np.linalg.pinv(stain_matrix).T)
    deconvolved = deconvolved.reshape((image.shape[0], image.shape[1], -1))
    return deconvolved

def reconstruct_rgb_from_channel(channel, stain_vector):
    """Reconstruct RGB image from a single channel and its stain vector."""
    od_channel = channel[:, :, np.newaxis] * stain_vector
    rgb_image = od_to_rgb(od_channel)
    return rgb_image

# Example stain matrix (for DAB and Hematoxylin)
# Adjust these values for your specific staining protocol
stain_matrix = np.array([
    [0.65, 0.70, 0.29],  # DAB
    [0.27, 0.70, 0.80],  # Hematoxylin
    [0.10, 0.25, 0.50]   # Residual/background
])

# Load the input image (replace 'path/to/image.jpg' with your file path)
input_image = cv2.imread('./input/3753-21C ACE-2 40X(2).jpg')

# Perform color deconvolution
deconvolved_channels = color_deconvolution(input_image, stain_matrix)

# Extract channels
dab_channel = deconvolved_channels[:, :, 0]
hematoxylin_channel = deconvolved_channels[:, :, 1]
residual_channel = deconvolved_channels[:, :, 2]

# Normalize channels to positive values
dab_channel = np.clip(dab_channel, 0, None)
hematoxylin_channel = np.clip(hematoxylin_channel, 0, None)
residual_channel = np.clip(residual_channel, 0, None)

# Reconstruct RGB images for each channel
dab_rgb = reconstruct_rgb_from_channel(dab_channel, stain_matrix[0])
hematoxylin_rgb = reconstruct_rgb_from_channel(hematoxylin_channel, stain_matrix[1])
residual_rgb = reconstruct_rgb_from_channel(residual_channel, stain_matrix[2])

# Plot and visualize
def plot_rgb_channels(original, dab_rgb, hematoxylin_rgb, residual_rgb):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    axs[1].imshow(cv2.cvtColor(dab_rgb, cv2.COLOR_BGR2RGB))
    axs[1].set_title("DAB RGB")
    axs[1].axis("off")
    
    axs[2].imshow(cv2.cvtColor(hematoxylin_rgb, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Hematoxylin RGB")
    axs[2].axis("off")
    
    axs[3].imshow(cv2.cvtColor(residual_rgb, cv2.COLOR_BGR2RGB))
    axs[3].set_title("Residual RGB")
    axs[3].axis("off")
    
    plt.tight_layout()
    plt.show()

plot_rgb_channels(input_image, dab_rgb, hematoxylin_rgb, residual_rgb)

