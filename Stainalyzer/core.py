
"""
core
----

Placeholder.

"""

############################################################################################################
### Import
############################################################################################################

import argparse

from Stainalyzer.trainer import Trainer

"""
import io
import os
import cv2
import sys
import math
import struct
import tempfile
import argparse
import numpy as np
import seaborn as sns
from math import ceil, log10
from time import perf_counter
from PIL import Image, ImageCms
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries

#from Stainalyzer.preprocessor import ImagePreprocessor
#from Stainalyzer.visualizations import Visualizations
#from Stainalyzer.dirichlet import DirichletProcessMixtureModel
#from Stainalyzer.filters import DABFilters
#from Stainalyzer.utils import ColorConverter
"""

############################################################################################################
### Constants
############################################################################################################

# Constants

"""
Image.MAX_IMAGE_PIXELS = 400_000_000  # Adjust this as needed
SEED = 1987
np.random.seed(SEED)
"""

############################################################################################################
### Argument Parsing
############################################################################################################

# Argument Parsing
def parse_args():
    """
    Argument Parsing.
    This function placeholder.

    Parameters:
    ----------
    placeholder : plchd
        The placeholder.

    Returns:
    --------
    Placeholder : plchd
        The placeholder.
     
    Raises
    ------
    Placeholder
        Placeholder. 
    """
    parser = argparse.ArgumentParser(description="Stainalyzer Tool")
    parser.add_argument("--input", required=True, help="Path to input file")
    return parser.parse_args()

############################################################################################################
### Core Functions
############################################################################################################

# Main Function
def main_function(input_training_folder, output_training_folder):
    """
    Main function.
    This function performs both training and testing of the model.

    Parameters:
    ----------
    input_training_folder : str
        The input folder containing training images.
    output_training_folder : str
        The output folder to record training objects.

    Returns:
    --------
    Placeholder : plchd
        The placeholder.
     
    Raises
    ------
    Placeholder
        Placeholder. 
    """

    # Model Training
    training_severity = 1.0
    trainer = Trainer(training_image_path=input_training_folder, severity=training_severity)
    trainer.train(output_location=output_training_folder)

# Entry Point
if __name__ == "__main__":

    # Input Output Files
    input_training_folder = "/Users/egg/Projects/Stainalyzer/data/DAB_Training/"
    output_training_folder = "/Users/egg/Projects/Stainalyzer/data/DAB_Training_Output"

    main_function(input_training_folder, output_training_folder)

"""

def main_function(input_colors_file_name, input_folder, input_training_folder, output_folder):
    "" "
    Main function.

    Parameters:
        input_colors_file_name (string): File containing community color names.
        input_folder (string): Folder containing TIFF images.
        input_training_folder (string): Path to DAB-only black-background files.
        output_folder (string): Path to store plots and files.
    "" "

    # Create results file
    output_file_name = os.path.join(output_folder, "results.txt")
    output_file = open(output_file_name, "w")

    # Iterate through kernel images
    for file_name in os.listdir(input_folder):

        # Tracking time
        time1 = perf_counter()

        # Join file path and print image name
        file_path = os.path.join(input_folder, file_name)
        output_file.write(f"# {file_name}\n")
        print(file_name)

        # If not an image in TIFF format continue to next
        if not file_name.lower().endswith(".tiff"):
            continue

        ########################################################
        # Step 1: Preprocessing
        ########################################################

        # Loading image
        image_preprocessor = ImagePreprocessor(file_path,
                                               replace_black_param=None,
                                               clahe_params=None,
                                               slic_params=None,
                                               kmeans_params=None
                                               )


        # Preprocessing
        image_preprocessor.preprocess()

        # Step 1: Tracking time - Preprocessing
        time12 = -1
        time1 = max(time1, time12)
        time2 = perf_counter()
        print(f"Pre-Processing: {time2-time1:.6f}")

        ########################################################
        # Step 2: CLAHE, CLIC and K-means Visualization
        ########################################################        

        "" "
        # Initializing visualizations
        viz = Visualizations(color_name_path=input_colors_file_name)

        # Visualize CLAHE-normalized image
        output_clahe_file_name = os.path.join(output_folder, "1_clahe.tiff")
        viz.save_clahe_image(image_preprocessor.processed_image, output_clahe_file_name,
                             color_space="LAB", output_format="TIFF")
        output_file.write(f"Dynamic Threshold: {image_preprocessor.dynamic_threshold:.2f}\n")

        # Visualize SLIC-segmentation image
        output_slic_file_name = os.path.join(output_folder, "2_slic.tiff")
        viz.save_superpixel_visualization(image_preprocessor.processed_image,
                                                     image_preprocessor.superpixel_segments,
                                                     output_slic_file_name,
                                                     color_space="LAB",
                                                     boundary_color=(255,0,0),
                                                     boundary_width=1,
                                                     title="SLIC Segmentation"
                                                     )

        # Visualize quantized image
        output_quant_file_name = os.path.join(output_folder, "3_quantized.tiff")
        viz.save_quantized_image(image_preprocessor.quantized_image,
                                            output_quant_file_name,
                                            color_space="LAB",
                                            output_format="TIFF"
                                            )
        output_file.write(f"Quantized Image Max Cluster Index: {np.max(image_preprocessor.quantized_image)}\n")
        output_file.write(f"Number of Centroids: {len(image_preprocessor.centroids)}\n")
        output_file.write(f"Total Pixels: {np.sum(image_preprocessor.pixel_counts)}\n")
        output_file.write(f"Image Pixels: {image_preprocessor.processed_image.shape[0] * image_preprocessor.processed_image.shape[1]}\n")

        # Visualize quantized & real image by cluster
        output_real_file_name = os.path.join(output_folder, "4_real_grid.tiff")
        real_output_path = os.path.join(output_folder, "4_real_grid")
        output_quant_file_name = os.path.join(output_folder, "5_quant_grid.tiff")
        quant_output_path = os.path.join(output_folder, "5_quant_grid")
        viz.plot_image_cut_by_cluster(image_preprocessor.processed_image,
                                                 image_preprocessor.cluster_labels,
                                                 image_preprocessor.centroids,
                                                 output_real_file_name,
                                                 real_output_path,
                                                 prefix="cluster_",
                                                 color_space="LAB",
                                                 output_format="TIFF"
                                                 )
        viz.plot_image_cut_by_cluster(image_preprocessor.quantized_image,
                                                 image_preprocessor.cluster_labels,
                                                 image_preprocessor.centroids,
                                                 output_quant_file_name,
                                                 quant_output_path,
                                                 prefix="cluster_",
                                                 color_space="LAB",
                                                 output_format="TIFF"
                                                 )

        # Visualize cluster colors image
        output_clustcol_file_name = os.path.join(output_folder, "6_cluster_colors.tiff")
        viz.plot_cluster_colors(image_preprocessor.centroids,
                                           image_preprocessor.pixel_counts,
                                           output_clustcol_file_name,
                                           color_space="LAB")

        # Step X: Tracking time - Placeholder
        time3 = perf_counter()
        print(f"Placeholder: {time3-time2:.6f}")
        "" "

        ########################################################
        # Step 3: DAB Color Distribution Estimation
        ########################################################  

        # Path to training images
        training_images_path = "/Users/egg/Projects/Stainalyzer/data/DAB_Training/"

        # Step 1: Read and clean training images (remove black background)
        def load_and_clean_training_images(path):
            training_images = []
            for i in range(1, 5):  # Assuming the training images are named Training1.png, Training2.png, etc.
                img = cv2.imread(f"{path}/Training{i}.tiff")
                if img is None:
                    raise ValueError(f"Image Training{i}.png not found in {path}.")
                
                # Remove black background
                mask = cv2.inRange(img, (0, 0, 0), (0, 0, 0))  # Mask for black background
                img[mask == 255] = [0, 0, 0]  # Ensure black background remains black (no interference)
                training_images.append(img)
            return training_images

        training_images = load_and_clean_training_images(training_images_path)

        # Step 2: Preprocess the training images
        def preprocess_training_images(images):
            preprocessed_data = []
            for img in images:

                # Preprocess image
                preprocessor = ImagePreprocessor(img)
                preprocessor.preprocess()

                # Collecting the processed data
                preprocessed_data.append({
                    "original_image": preprocessor.original_image,
                    "processed_image": preprocessor.processed_image,
                    "quantized_image": preprocessor.quantized_image,
                    "centroids": preprocessor.centroids,
                    "cluster_labels": preprocessor.cluster_labels
                })
            return preprocessed_data

        training_preprocessed = preprocess_training_images(training_images)

        # Step 3: Create the DABDistribution
        dab_distribution = DABDistribution()
        for data in training_preprocessed:

            # Assuming quantized_image contains LAB pixel data
            lab_pixels = cv2.cvtColor(data["quantized_image"], cv2.COLOR_BGR2LAB).reshape(-1, 3)
            dab_distribution.add_data(lab_pixels)

        # Step 4: Collect test image data
        test_clahe_image = image_preprocessor.processed_image
        test_quantized_image = image_preprocessor.quantized_image
        test_centroids = image_preprocessor.centroids
        test_labels = image_preprocessor.cluster_labels

        # Step 5: Create a MultivariateLABDistribution for each test image cluster
        test_distributions = []
        for cluster_id in range(len(test_centroids)):

            mask = test_labels == cluster_id  # Create a mask for the cluster
            cluster_pixels = test_clahe_image[mask]  # Extract pixels for the cluster
            
            # No need for cvtColor; test_clahe_image is already in LAB
            lab_pixels = cluster_pixels.reshape(-1, 3)
            
            cluster_distribution = MultivariateLABDistribution()
            cluster_distribution.fit(lab_pixels, n_components=3)
            test_distributions.append(cluster_distribution)

        #output_dabdist_image_2d = os.path.join(output_folder, f"dab_distribution_2d.tiff")
        #dab_distribution.dab_distribution.plot(projection='2D', output_file=output_dabdist_image_2d)
        #output_dabdist_image_3d = os.path.join(output_folder, f"dab_distribution_3d.tiff")
        #dab_distribution.dab_distribution.plot(projection='3D', output_file=output_dabdist_image_3d)

        # Step 6: Calculate distances between test clusters and DABDistribution
        for i, cluster_distribution in enumerate(test_distributions):
            output_file.write(f"{dab_distribution.dab_distribution}\n")
            output_file.write(f"{cluster_distribution}\n")
            distance = dab_distribution.dab_distribution.distance_to(cluster_distribution, metric="Wasserstein")
            output_file.write(f"Cluster {i}: Distance to DAB distribution = {distance:.4f}\n")
            output_file.write(f"########################################################\n\n")

        # Step X: Tracking time - Placeholder
        #time4 = perf_counter()
        #print(f"Placeholder: {time4-time3:.6f}")
        # Step X: Tracking time - Placeholder
        #time5 = perf_counter()
        #print(f"Placeholder: {time5-time4:.6f}")
        # Step X: Tracking time - Placeholder
        #time6 = perf_counter()
        #print(f"Placeholder: {time6-time5:.6f}")
        # Step X: Tracking time - Placeholder
        #time7 = perf_counter()
        #print(f"Placeholder: {time7-time6:.6f}")
        # Step X: Tracking time - Placeholder
        #time8 = perf_counter()
        #print(f"Placeholder: {time8-time7:.6f}")

        # Step 4.8: Plot cluster colors in the real image
        #output_clusters_in_quantized_image = os.path.join(output_folder, f"{file_name}_clusters_in_quantized.tiff")
        #plot_clusters_in_quantized_image(lab_image, quantized_image, output_clusters_in_quantized_image, color_space="LAB")

        # Step 4.9: Tracking time - 
        #time9 = perf_counter()
        #print(f"8,9: {time9-time8:.6f}")

        # Step 4.10: Save the plot of cluster colors
        #plot_cluster_colors_file = os.path.join(output_folder, f"{file_name}_cluster_colors.tiff")
        #plot_cluster_colors(centroids, pixel_counts, hexadecimal_to_name, plot_cluster_colors_file)

        # Step 4.11: Tracking time - 
        #time10 = perf_counter()
        #print(f"9,10: {time10-time9:.6f}")

        # Step 4.12: Plot cluster colors in the real image with Centroid Color Name annotation
        #output_annotated_color_name_image = os.path.join(output_folder, f"{file_name}_annotated_color_name.tiff")
        #plot_annotated_clusters_in_quantized_image(image, quantized_image, centroids, hexadecimal_to_name, output_annotated_color_name_image)

        # Step 4.13: Tracking time - 
        #time11 = perf_counter()
        #print(f"10,11: {time11-time10:.6f}")

        # GAUSSIAN

        #time12 = perf_counter()
        #print(f"11,12: {time12-time11:.6f}")

        # Step 4.12: Write RGB/HSV/LAB distributions to the file
        #distributions = calculate_color_distributions(rgb_image, hsv_image, lab_image, quantized_image, centroids,
        #                                              pixel_counts, hexadecimal_to_name, output_file)

    # Closing file    
    output_file.close()


def test_function(input_folder, test_name, output_folder):

    def save_images_side_by_side(img1, img2, output_path, title1="Image 1", title2="Image 2"):
        "" "
        Save two images side by side, resizing them to have the same dimensions.

        Parameters:
            img1 (numpy.ndarray): First image in BGR format.
            img2 (numpy.ndarray): Second image in BGR format.
            output_path (str): Path to save the output image.
            title1 (str): Title for the first image. Default is "Image 1".
            title2 (str): Title for the second image. Default is "Image 2".
        "" "
        # Ensure images are in BGR format; convert to RGB for matplotlib
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Resize both images to the same size (use the dimensions of the first image)
        height, width = img1_rgb.shape[:2]
        img2_resized = cv2.resize(img2_rgb, (width, height), interpolation=cv2.INTER_AREA)
        
        # Create a figure to display the images
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
        
        # Display the first image
        axes[0].imshow(img1_rgb)
        axes[0].axis("off")
        axes[0].set_title(title1)
        
        # Display the second image
        axes[1].imshow(img2_resized)
        axes[1].axis("off")
        axes[1].set_title(title2)
        
        # Save the combined plot as an image
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)  # Save with high resolution
        plt.close()  # Close the figure to free up memory


    # Load the image
    input_path = os.path.join(input_folder, test_name)
    img = cv2.imread(input_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_img = ColorConverter.rotate_hsv(hsv_img, rotation=90)

    # Define color ranges
    _0_20_range = (np.array([1, 0, 0]), np.array([20, 255, 255]))
    _20_40_range = (np.array([20, 0, 0]), np.array([40, 255, 255]))
    _40_60_range = (np.array([40, 0, 0]), np.array([60, 255, 255]))
    _60_80_range = (np.array([60, 0, 0]), np.array([80, 255, 255]))
    _80_100_range = (np.array([80, 0, 0]), np.array([100, 255, 255]))
    _100_120_range = (np.array([100, 0, 0]), np.array([120, 255, 255]))
    _120_140_range = (np.array([120, 0, 0]), np.array([140, 255, 255]))
    _140_160_range = (np.array([140, 0, 0]), np.array([160, 255, 255]))
    _160_180_range = (np.array([160, 0, 0]), np.array([180, 255, 255]))

    # Create masks
    mask1 = cv2.inRange(hsv_img, _0_20_range[0], _0_20_range[1])
    mask2 = cv2.inRange(hsv_img, _20_40_range[0], _20_40_range[1])
    mask3 = cv2.inRange(hsv_img, _40_60_range[0], _40_60_range[1])
    mask4 = cv2.inRange(hsv_img, _60_80_range[0], _60_80_range[1])
    mask5 = cv2.inRange(hsv_img, _80_100_range[0], _80_100_range[1])
    mask6 = cv2.inRange(hsv_img, _100_120_range[0], _100_120_range[1])
    mask7 = cv2.inRange(hsv_img, _120_140_range[0], _120_140_range[1])
    mask8 = cv2.inRange(hsv_img, _140_160_range[0], _140_160_range[1])
    mask9 = cv2.inRange(hsv_img, _160_180_range[0], _160_180_range[1])

    hsv_img = ColorConverter.rotate_hsv(hsv_img, rotation=90)

    # Apply the mask to the original image
    pixels1 = cv2.bitwise_and(hsv_img, hsv_img, mask=mask1)
    pixels2 = cv2.bitwise_and(hsv_img, hsv_img, mask=mask2)
    pixels3 = cv2.bitwise_and(hsv_img, hsv_img, mask=mask3)
    pixels4 = cv2.bitwise_and(hsv_img, hsv_img, mask=mask4)
    pixels5 = cv2.bitwise_and(hsv_img, hsv_img, mask=mask5)
    pixels6 = cv2.bitwise_and(hsv_img, hsv_img, mask=mask6)
    pixels7 = cv2.bitwise_and(hsv_img, hsv_img, mask=mask7)
    pixels8 = cv2.bitwise_and(hsv_img, hsv_img, mask=mask8)
    pixels9 = cv2.bitwise_and(hsv_img, hsv_img, mask=mask9)

    # Convert back to RGB
    image1 = cv2.cvtColor(pixels1, cv2.COLOR_HSV2BGR)
    image2 = cv2.cvtColor(pixels2, cv2.COLOR_HSV2BGR)
    image3 = cv2.cvtColor(pixels3, cv2.COLOR_HSV2BGR)
    image4 = cv2.cvtColor(pixels4, cv2.COLOR_HSV2BGR)
    image5 = cv2.cvtColor(pixels5, cv2.COLOR_HSV2BGR)
    image6 = cv2.cvtColor(pixels6, cv2.COLOR_HSV2BGR)
    image7 = cv2.cvtColor(pixels7, cv2.COLOR_HSV2BGR)
    image8 = cv2.cvtColor(pixels8, cv2.COLOR_HSV2BGR)
    image9 = cv2.cvtColor(pixels9, cv2.COLOR_HSV2BGR)

    # Visualize or save the masks
    output_filter1 = os.path.join(output_folder, "01.png")
    save_images_side_by_side(img, image1, output_filter1, "Normal", "XXX")
    output_filter2 = os.path.join(output_folder, "02.png")
    save_images_side_by_side(img, image2, output_filter2, "Normal", "XXX")
    output_filter3 = os.path.join(output_folder, "03.png")
    save_images_side_by_side(img, image3, output_filter3, "Normal", "XXX")
    output_filter4 = os.path.join(output_folder, "04.png")
    save_images_side_by_side(img, image4, output_filter4, "Normal", "XXX")
    output_filter5 = os.path.join(output_folder, "05.png")
    save_images_side_by_side(img, image5, output_filter5, "Normal", "XXX")
    output_filter6 = os.path.join(output_folder, "06.png")
    save_images_side_by_side(img, image6, output_filter6, "Normal", "XXX")
    output_filter7 = os.path.join(output_folder, "07.png")
    save_images_side_by_side(img, image7, output_filter7, "Normal", "XXX")
    output_filter8 = os.path.join(output_folder, "08.png")
    save_images_side_by_side(img, image8, output_filter8, "Normal", "XXX")
    output_filter9 = os.path.join(output_folder, "09.png")
    save_images_side_by_side(img, image9, output_filter9, "Normal", "XXX")

    sys.exit(0)

    # Clahe play
    img = cv2.imread(input_path)
    out_folder = os.path.join(output_folder, "9_clahe")

    def plot_clahe(img, clipLimit, tileGridSize, output_folder, output_name):
        output_clahe = os.path.join(output_folder, output_name)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        hsv_img[..., 2] = clahe.apply(hsv_img[..., 2])
        clahe_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        save_images_side_by_side(img, clahe_img, output_clahe, "Normal", "CLAHE")

    # Print multiple
    clipvec = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    gridvec = [(8, 8), (12, 12), (16, 16), (20, 20), (30, 30), (40, 40), (50, 50), (80, 80)]
    for clipy in clipvec:
        for gridy in gridvec:
            output_name = f"9_clahe_{int(clipy)}_{int(gridy[0])}.png"
            #plot_clahe(img, clipy, gridy, out_folder, output_name)

    # Gaussian Blur Play
    img = cv2.imread(input_path)
    out_folder = os.path.join(output_folder, "10_gauss")

    def plot_gauss(img, ksize, sigmaX, output_folder, output_name):
        output_gauss = os.path.join(output_folder, output_name)
        blurred_img = cv2.GaussianBlur(img, ksize, sigmaX)
        save_images_side_by_side(img, blurred_img, output_gauss, "Normal", "GBlur")

    # Print multiple
    ksize = [(5, 5), (9, 9), (13, 13), (25, 25)]
    sigmaX = [0, 1, 2, 3, 10]
    for kk in ksize:
        for sig in sigmaX:
            output_name = f"10_gauss_{int(kk[0])}_{int(sig)}.png"
            #plot_gauss(img, kk, sig, out_folder, output_name)

    # Median Blur
    img = cv2.imread(input_path)
    out_folder = os.path.join(output_folder, "11_median")

    def plot_median(img, ksize, output_folder, output_name):
        output_median = os.path.join(output_folder, output_name)
        median_blurred_img = cv2.medianBlur(img, ksize)
        save_images_side_by_side(img, median_blurred_img, output_median, "Normal", "MBlur")

    # Print multiple
    ksize = [3, 5, 9, 13, 15, 19, 25]
    for kk in ksize:
        output_name = f"11_median_{int(kk)}.png"
        #plot_median(img, kk, out_folder, output_name)

    # HSV JustH Filter
    def h_quantization(image, svalue):
        "" "
        Quantizes the saturation (S) and value (V) channels of an image to a fixed value, leaving hue (H) intact.

        Parameters:
            image (numpy.ndarray): Input image in BGR format (as read by OpenCV).
            svalue (int): The value to set for both S (saturation) and V (value) channels. Must be in range [0, 255].

        Returns:
            numpy.ndarray: The resulting image with modified S and V channels, converted back to BGR format.
        "" "
        if not (0 <= svalue <= 255):
            raise ValueError("svalue must be in the range [0, 255].")
        
        # Convert the input image from BGR to HSV
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Set all S (saturation) and V (value) channels to the given svalue
        hsv_img[:, :, 1] = svalue  # Modify the S channel
        hsv_img[:, :, 2] = svalue  # Modify the V channel

        # Convert the modified HSV image back to BGR
        result_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        return result_img

    # HSV Quantization
    img = cv2.imread(input_path)
    out_folder = os.path.join(output_folder, "12_hsv")

    def plot_hsvquant(img, svalue, output_folder, output_name):
        output_hsvquant = os.path.join(output_folder, output_name)
        hsvquant_img = h_quantization(img, svalue)
        save_images_side_by_side(img, hsvquant_img, output_hsvquant, "Normal", "HSV_Quant")

    # Print multiple
    svector = [0, 1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 254, 255]
    for ss in svector:
        output_name = f"12_hsv_{int(ss)}.png"
        #plot_hsvquant(img, ss, out_folder, output_name)

    # Color Quantization
    img = cv2.imread(input_path)
    out_folder = os.path.join(output_folder, "13_KMquant")

    def plot_kmquant(img, kmean, maxiter, epsilon, attempty, output_folder, output_name):
        output_kmquant = os.path.join(output_folder, output_name)
        data = img.reshape((-1, 3)).astype(np.float32)
        _, labels, centers = cv2.kmeans(data, kmean, None, 
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                  maxiter,
                                                  epsilon
                                                  ), 
                                        attempts=attempty,
                                        flags=cv2.KMEANS_RANDOM_CENTERS
                                        )
        kmquant_img = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)
        save_images_side_by_side(img, kmquant_img, output_kmquant, "Normal", "KM_Quant")

    # Print multiple
    kVec = [3, 4, 5, 6, 7, 8, 9, 10]
    maxiterVec = [100]
    epsilonVec = [0.1]
    attemptsVec = [100]
    for kk in kVec:
        for maxiter in maxiterVec:
            for epsilon in epsilonVec:
                for attempty in attemptsVec:
                    output_name = f"13_KM_{int(kk)}_{maxiter}_{int(epsilon*10)}_{attempty}.png"
                    #plot_kmquant(img, kk, maxiter, epsilon, attempty, out_folder, output_name)

    # Dirichlet Process Mixture Model (DPMM Quantization)
    img = cv2.imread(input_path)
    out_folder = os.path.join(output_folder, "14_DPMMquant")

    # Function for Dirichlet Process Mixture Model quantization
    def plot_dpmm_quant(img, n_components, max_iter, tol, n_init, output_folder, output_name):
        output_path = os.path.join(output_folder, output_name)
        dimensions= [0, 1, 2]
        data = img.reshape((-1, 3))[:, dimensions]
        dpmm = DirichletProcessMixtureModel(
                                            n_components=n_components,
                                            covariance_type='full',
                                            random_state=SEED,
                                            max_iter=max_iter,
                                            tol=tol,
                                            n_init=n_init,
                                            init_params='kmeans',
                                            weight_concentration_prior=None,
                                            reg_covar=1e-6,
                                            verbose=0
                                            )
        dpmm.fit(data)
        quantized_img = dpmm.quantize_image(img, dimensions)
        save_images_side_by_side(img, quantized_img, output_path, "Normal", "DPMM_Quant")

    # Parameter vectors for testing
    n_components_vec = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]  # Number of components
    maxiter_vec = [200]  # Covariance type
    tolVec = [1e-3]
    ninitvec = [2]
    # Perform grid-search-like testing
    for n_components in n_components_vec:
        for maxiter in maxiter_vec:
            for tol in tolVec:
                for ninit in ninitvec:
                    output_name = f"14_DPMMquant_{n_components}_{maxiter}_{abs(log10(tol))}_{ninit}.png"
                    #plot_dpmm_quant(img, n_components, maxiter, tol, ninit, out_folder, output_name)

    # Edge Detection
    img = cv2.imread(input_path)
    out_folder = os.path.join(output_folder, "15_canny_edges")

    # Create output folder if it doesn't exist
    os.makedirs(out_folder, exist_ok=True)

    # Function to apply Canny edge detection for a single parameter combination
    def plot_canny(img, threshold1, threshold2, apertureSize, L2gradient, output_folder, output_name):
        "" "
        Applies Canny edge detection with given parameters and saves the result side-by-side with the original image.
        
        Parameters:
            img (np.ndarray): Input image.
            threshold1 (float): First threshold for the hysteresis procedure.
            threshold2 (float): Second threshold for the hysteresis procedure.
            apertureSize (int): Aperture size for the Sobel operator. Must be odd (e.g., 3, 5, 7).
            L2gradient (bool): Whether to use a more accurate L2 norm for gradient magnitude.
            output_folder (str): Output directory to save the result.
            output_name (str): Name of the output file.
        "" "
        output_path = os.path.join(output_folder, output_name)
        edges = cv2.Canny(img, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient)
        save_images_side_by_side(img, edges, output_path, "Original", "Canny Edges")

    # Defining vectors of parameters for testing
    threshold1_vec = [10, 25, 50, 75]  # First threshold for hysteresis
    threshold2_vec = [100, 125, 150]  # Second threshold for hysteresis
    apertureSize_vec = [3, 5]  # Aperture size for the Sobel operator
    L2gradient_vec = [False]  # Whether to use L2 norm for gradient magnitude

    # Nested loops to test all parameter combinations
    for t1 in threshold1_vec:
        for t2 in threshold2_vec:
            for aperture in apertureSize_vec:
                for l2grad in L2gradient_vec:
                    output_name = f"15_canny_t1-{t1}_t2-{t2}_ap-{aperture}_l2-{int(l2grad)}.png"
                    #plot_canny(img, t1, t2, aperture, l2grad, out_folder, output_name)


    # Histogram Equalization
    img = cv2.imread(input_path)
    out_folder = os.path.join(output_folder, "16_hist_eq")

    # Create output folder if it doesn't exist
    os.makedirs(out_folder, exist_ok=True)

    # Function to apply histogram equalization for a single parameter combination
    def plot_hist_eq(img, clahe_clip_limit, clahe_tile_grid_size, output_folder, output_name):
        "" "
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and saves the result side-by-side with the original image.

        Parameters:
            img (np.ndarray): Input image.
            clahe_clip_limit (float): Threshold for contrast limiting in CLAHE.
            clahe_tile_grid_size (tuple): Size of grid for histogram equalization.
            output_folder (str): Output directory to save the result.
            output_name (str): Name of the output file.
        "" "
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        eq_img = clahe.apply(gray)  # Apply histogram equalization
        save_images_side_by_side(gray, eq_img, os.path.join(output_folder, output_name), "Original", "Histogram Equalized")

    # Defining vectors of parameters for testing
    clahe_clip_limit_vec = [2.0, 3.0, 4.0, 5.0]  # Threshold for contrast limiting
    clahe_tile_grid_size_vec = [(8, 8), (16, 16), (32, 32)]  # Grid sizes for local equalization

    # Nested loops to test all parameter combinations
    for clip_limit in clahe_clip_limit_vec:
        for tile_grid_size in clahe_tile_grid_size_vec:
            output_name = f"16_hist_eq_{clip_limit}_{tile_grid_size[0]}.png"
            #plot_hist_eq(img, clip_limit, tile_grid_size, out_folder, output_name)

    def sigmoid_function(x, gain=1, midpoint=128):
        "" "
        Applies the sigmoid function to the input.

        Parameters
        ----------
        x : np.ndarray or float
            The input value(s) to apply the sigmoid function to.
        gain : float, optional
            The steepness of the sigmoid curve. Default is 1.
        midpoint : float, optional
            The midpoint value where the sigmoid transitions (center of the curve). Default is 128.

        Returns
        -------
        np.ndarray or float
            The transformed value(s) after applying the sigmoid function.
        "" "
        return 1 / (1 + np.exp(-gain * (x - midpoint)))

    # Initialize image and path
    img = cv2.imread(input_path)
    out_folder1 = os.path.join(output_folder, "17_dabf_compress")
    out_folder2 = os.path.join(output_folder, "18_dabf_blur")
    out_folder3 = os.path.join(output_folder, "19_dabf_nonlinear")
    os.makedirs(out_folder1, exist_ok=True)
    os.makedirs(out_folder2, exist_ok=True)
    os.makedirs(out_folder3, exist_ok=True)

    # Load DABFilters class (already implemented)
    dab_filter = DABFilters(img, mode="HSV")

    # Function to apply "compress_sv"
    def plot_compress_sv(img, alpha, beta, output_folder, output_name):
        dab_filter.set_image(img)  # Set the image in the filter
        compressed_img = dab_filter.compress_sv(alpha=alpha, beta=beta)
        save_images_side_by_side(img, compressed_img, os.path.join(output_folder, output_name), "Original", "Compressed SV")

    # Function to apply "blur_sv"
    def plot_blur_sv(img, kernel_size, sigmaX, sigmaY, output_folder, output_name):
        dab_filter.set_image(img)  # Set the image in the filter
        blurred_img = dab_filter.blur_sv(kernel_size=kernel_size, sigmaX=sigmaX, sigmaY=sigmaY)
        save_images_side_by_side(img, blurred_img, os.path.join(output_folder, output_name), "Original", "Blurred SV")

    # Function to apply "nld_sv"
    def plot_nld_sv(img, nonlinear_function, output_folder, output_name):
        dab_filter.set_image(img)  # Set the image in the filter
        nonlinear_img = dab_filter.nld_sv(nonlinear_function=nonlinear_function)
        save_images_side_by_side(img, nonlinear_img, os.path.join(output_folder, output_name), "Original", "NLD SV")

    # Defining parameter vectors for each filter
    # Compress SV parameters
    alpha_vec = [0.1, 0.3, 0.5, 0.7, 1.0]
    beta_vec = [0.1, 0.3, 0.5, 0.7, 1.0]

    # Blur SV parameters
    kernel_size_vec = [(5, 5), (11, 11)]
    sigmaX_vec = [2, 5, 10]
    sigmaY_vec = [2, 5, 10]

    # NLD SV parameters
    gain_vec = [1, 5, 10]
    midpoint_vec = [0.25, 0.5, 0.75]

    # Nested loops to test all parameter combinations
    # Testing "compress_sv"
    for alpha in alpha_vec:
        for beta in beta_vec:
            output_name = f"17_dabf_compress_{alpha}_{beta}.png"
            #plot_compress_sv(img, alpha, beta, out_folder1, output_name)

    # Testing "blur_sv"
    for kernel_size in kernel_size_vec:
        for sigmaX in sigmaX_vec:
            for sigmaY in sigmaY_vec:
                output_name = f"18_dabf_blur_{kernel_size[0]}_{sigmaX}_{sigmaY}.png"
                #plot_blur_sv(img, kernel_size, sigmaX, sigmaY, out_folder2, output_name)

    # Testing "nld_sv"
    for gain in gain_vec:
        for midpoint in midpoint_vec:
            def sigmoid_function(x):
                return 1 / (1 + np.exp(-gain * (x - midpoint)))

            output_name = f"19_dabf_nonlinear_{int(gain*100)}_{int(midpoint*100)}.png"
            #plot_nld_sv(img, sigmoid_function, out_folder3, output_name)

if __name__ == "__main__":

    # Input Output Files
    #input_colors_file_name = "/Users/egg/Projects/Stainalyzer/data/colornames.txt"
    #input_folder = "/Users/egg/Projects/Stainalyzer/examples/input/tiff/"
    #input_training_folder = "/Users/egg/Projects/Stainalyzer/examples/output/hypertraining1/"
    #output_folder = "/Users/egg/Projects/Stainalyzer/examples/output/hypertraining1/"
    #os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

    # Calling main
    #main_function(input_colors_file_name, input_folder, input_training_folder, output_folder)

    # Tests
    #input_folder = "/Users/egg/Projects/Stainalyzer/data/input_test/"
    #test_name = "test.tiff"
    #output_folder = "/Users/egg/Projects/Stainalyzer/data/output_test/"
    #test_function(input_folder, test_name, output_folder)

"""

"""
1. Annotation Removal: Change pure black pixels to its closest pixel (annotation elements on the Figure) (image in RGB)
2. Pre-processing: Pre-Threshold Normalization: CLAHE in LAB for Normalized-LAB
2. Thresholding: Calculate dynamic threshold T
3. SLIC: Perform SLIC with 10 * T (image in Normalized-LAB) and T compactness

4. Clustering: Use SLIC to initialize K-means with Weighted Centroids Initialization, using number of clusters = T (image in Normalized-LAB)
5. Pre-KDE Normalization: Image Normalization Before Distribution Calculation for: RGB, HSV, LAB, LUV & XYZ.
6. Calculate the KDE-Distribution of pixels for: RGB, HSV, LAB, LUV & XYZ.
7. Calculate accurate & weighted distances between centroid/KDE-Distributions and DAB-brown for: RGB, HSV, LAB, LUV & XYZ.
8. Post-Processing: Morphological operations for refined cluster boundaries.
9. Visualization: Overlay results for manual inspection.

A. Pre-Threshold Normalization with CLAHE: [Step 1 above]
A.1. Normalize the image intensity or brightness before calculating T. Differences in staining quality and lighting can cause inconsistencies.
B. Weighted Centroids Initialization: [Step 5 above]
B.1. Weight the centroids using the distribution of pixel intensity within each SLIC superpixel. This improves the representation of stained regions.
C. Image Normalization Before Distribution Calculation: [Step 6 above]
C.1. Normalize each color channel within its valid range before calculating distributions. This ensures consistency when comparing across images.
D. Use Kernel Density Estimation (KDE) Distributions: [Step 7 above]
D.1. Replace simple mean/std with non-parametric KDE to better approximate real distributions of pixel intensities:
E. Calculate distances: [Step 8 above]
E.1. Centroid and DAB-brown distribution:
E.1.1. Perceptual Weights for LAB (Improvement):
E.1.2. Apply perceptual weights to LAB distances. For example:
E.1.3. L has less perceptual importance than A and B. Scale distances by perceptual importance (e.g., L_weight=0.5, A_weight=1, B_weight=1).
E.1.4. Replace Euclidean distances with Mahalanobis distance or Earth Mover’s Distance (EMD).
E.2. Real image distribution and DAB-brown distribution
E.2.1. Full Distribution Comparison (Improvement):
E.2.2. Use Wasserstein distance (1D EMD) for comparing real distributions rather than centroid-based Euclidean measures. This is crucial if real distributions are not Gaussian.
E.3. Quantized image distribution and DAB-brown distribution
E.3.1. Bias Reduction in Quantized Distribution (NEW):
E.3.2. When calculating distances, consider the bias introduced by quantization:
E.3.3. Quantized distributions often lose variance, making distances smaller than they should be.
E.3.4. Weight quantized distributions by the pixel count in each cluster.

"""




