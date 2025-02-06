
# Try to perform UMAP + HDBSCAN clustering

# Import packages
import os
import cv2
import umap
import hdbscan
import numpy as np
import matplotlib.pyplot as plt

# Input Output Files
input_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/"
output_folder = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/"
output_file_name = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/6520-21E.txt"
output_file = open(output_file_name, "w")

# Iterate through kernel images
for file_name in os.listdir(input_folder):

    # Join file path
    file_path = os.path.join(input_folder, file_name)

    # If not an image, continue
    if(os.path.splitext(file_path)[-1] not in [".png", ".jpg"]):
        continue

    # Load the RGB image
    image = cv2.imread(file_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB to LAB color space for CLAHE
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])

    # Convert back to RGB color space
    enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    # Save the original and enhanced images for comparison
    clahe_image_file_name = os.path.join(output_folder, "6520-21E_CLAHE.png")
    cv2.imwrite(clahe_image_file_name, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))

    ########

    # Step 1: Load Image and Flatten Pixels
    image = cv2.imread(clahe_image_file_name)  # Replace with your file path
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = rgb_image.reshape(-1, 3)  # Flatten image to (num_pixels, 3)

    # Step 2: Apply UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = umap_reducer.fit_transform(pixels)

    # Step 3: Clustering with HDBSCAN
    hdbscan_clusterer = hdbscan.HDBSCAN(min_samples=50, min_cluster_size=50)
    clusters = hdbscan_clusterer.fit_predict(reduced)

    # Step 4: Visualize Reduced Space
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='tab10', s=1)
    plt.colorbar(label="Cluster Labels")
    plt.title("UMAP Clustering with HDBSCAN")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    hdbscan_image_file_name = os.path.join(output_folder, "6520-21E_HDBSCAN.png")
    plt.savefig(hdbscan_image_file_name, dpi=300, bbox_inches="tight")

    # Step 5: Map Clusters to Original Image
    clustered_image = clusters.reshape(rgb_image.shape[:2])  # Reshape back to image dimensions

    # Plot the regions colored by clusters
    plt.figure(figsize=(10, 10))
    plt.imshow(clustered_image, cmap='tab10')
    plt.title("Clustered Regions in the Image")
    plt.axis("off")
    original_image_clustered_file_name = os.path.join(output_folder, "6520-21E_FINAL.png")
    plt.savefig(original_image_clustered_file_name, dpi=300, bbox_inches="tight")

    # Optional: Print unique clusters found
    unique_clusters = np.unique(clusters)
    print(f"Unique clusters found (including noise cluster '-1'): {unique_clusters}")

# Closing file    
output_file.close()














