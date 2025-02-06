import os
import shutil

def copy_images_from_file(image_list_file, destination_dir):
    """
    Copies image files listed in a file to a specified destination directory.

    Parameters:
    -----------
    image_list_file : str
        Path to the file containing image paths, one per line (may contain spaces).
    destination_dir : str
        Path to the directory where images will be copied.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    with open(image_list_file, "r", encoding="utf-8") as file:
        for line in file:
            image_path = line.strip()
            if os.path.exists(image_path):
                try:
                    shutil.copy(image_path, destination_dir)
                    print(f"Copied: {image_path}")
                except Exception as e:
                    print(f"Failed to copy {image_path}: {e}")
            else:
                print(f"File not found: {image_path}")

if __name__ == "__main__":
    image_list_file = "/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle.txt"
    destination_dir = "/Users/egg/Projects/Stainalyzer/data/selected_controle"
    copy_images_from_file(image_list_file, destination_dir)
