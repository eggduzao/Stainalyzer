# Mac OS command to count all image files in a directory tree:
# find /Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle -type f | grep -E '\.(jpg|jpeg|png|gif|bmp|tiff)$' | wc -l
# find /Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento -type f | grep -E '\.(jpg|jpeg|png|gif|bmp|tiff)$' | wc -l

import os
import cv2
import time
import glob

def find_images(root_dir):
    """
    Recursively find all image files in a directory.

    Parameters:
    -----------
    root_dir : str
        The root directory to search for image files.

    Returns:
    --------
    list
        A list of image file paths.
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    image_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files


def display_images(image_files, display_time=0.5, output_file="selected_images.txt"):
    """
    Display images in a loop and allow user to mark them.

    Parameters:
    -----------
    image_files : list
        List of image file paths to display.
    display_time : float, optional
        Time in seconds to display each image (default is 0.5).
    output_file : str, optional
        Path to the file where selected images will be saved (default is "selected_images.txt").
    """
    selected_images = []

    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Resize to 50% of original size
        height, width = image.shape[:2]
        image = cv2.resize(image, (width // 2, height // 2))

        cv2.imshow("Image Viewer", image)
        start_time = time.time()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                selected_images.append(image_path)
                break
            elif key == ord("q"):
                # Save selected images before quitting
                with open(output_file, "w") as f:
                    for img in selected_images:
                        f.write(img + "\n")
                cv2.destroyAllWindows()
                return
            elif time.time() - start_time >= display_time:
                break
    
    # Save selected images at the end of the loop
    with open(output_file, "w") as f:
        for img in selected_images:
            f.write(img + "\n")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    root_directory = "/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle"
    output_file_name = "/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle.txt"
    #root_directory = "/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento"
    #output_file_name = "/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento.txt"
    images = find_images(root_directory)
    if not images:
        print("No images found.")
    else:
        display_images(images, display_time=0.75, output_file=output_file_name)
