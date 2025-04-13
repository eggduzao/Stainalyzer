
"""
imagetype_detector
"""

############################################################################################################
### Import
############################################################################################################

import os
import cv2
import psutil
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

############################################################################################################
### Constants
############################################################################################################

# Constants
SEED = 1987
np.random.seed(SEED)

############################################################################################################
### Classes
############################################################################################################

class ImagetypeDetector:
    """
    ImagetypeDetector TODO DESCRIPTION
    """

    def __init__(self, image_path=None):
        """
        Placeholder
        """

        # Image directory
        self.image_path = Path(image_path)

        # Supported extensions
        self.supported_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    # Color range definitions for H&E and DAB heuristics
    def is_he_like(self, image):
        # Heuristic for pink-purple H&E
        avg_color = image.mean(axis=(0, 1))
        return avg_color[0] > 100 and avg_color[2] > 100 and avg_color[1] < 90  # R + B high, G low

    def is_dab_like(self, image):
        # Brownish DAB (stains cell cytoplasm/bodies)
        avg_color = image.mean(axis=(0, 1))
        return avg_color[0] > 100 and avg_color[1] > 80 and avg_color[2] < 80  # R + G > B

    def is_grayscale(self, img):
        return len(img.shape) == 2 or (img.shape[2] == 3 and np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2]))

    # Classify image based on simple properties
    def classify_image(self, image_path):

        img = cv2.imread(image_path)
        if img is None:
            return "Unreadable"
        
        height, width = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]

        fname = os.path.basename(image_path).lower()

        # Check grayscale (fluorescence?)
        if self.is_grayscale(img):
            if "cycif" in fname or "multiplex" in fname:
                return "Fluorescence > Multiplexed"
            elif "vectra" in fname:
                return "Fluorescence > Brightfield (Vectra)"
            else:
                return "Fluorescence > Other"

        # RGB images
        if self.is_he_like(img):
            return "Brightfield > H&E"
        elif self.is_dab_like(img):
            return "Brightfield > DAB"
        elif "phase" in fname:
            return "Brightfield > Phase-contrast"
        elif "dic" in fname:
            return "Brightfield > DIC"
        elif "ihc" in fname or "cihc" in fname:
            return "Brightfield > cIHC"
        elif "confocal" in fname:
            return "Confocal"
        elif "mibi" in fname or "imc" in fname or "tof" in fname:
            return "Mass Spectrometry Imaging"
        else:
            return "Brightfield > Regular"

    # Scan folder
    def scan_folder(self):
        counts = defaultdict(int)
        results = {}
        for root, _, files in os.walk(self.image_path):
            for fname in files:
                if fname.lower().endswith(self.supported_exts):
                    fpath = os.path.join(root, fname)
                    label = self.classify_image(fpath)
                    counts[label] += 1
                    results[fpath] = label
        return counts, results

# Run and print summary
if __name__ == "__main__":

    # image_path = Path("/Users/egg/Desktop/data-science-bowl-2018/stage1_test")
    # Fluorescence > Other: 53
    # Brightfield > Regular: 12
    # image_path = Path("/Users/egg/Desktop/data-science-bowl-2018/stage1_train")
    # Fluorescence > Other: 30023
    # Brightfield > Regular: 106
    # Brightfield > H&E: 2
    # image_path = Path("/Users/egg/Desktop/data-science-bowl-2018/stage2_test_final")
    # Fluorescence > Other: 2800
    # Brightfield > Regular: 215
    # Brightfield > H&E: 4
    imd = ImagetypeDetector(image_path)
    counts, image_results = imd.scan_folder()
    print("\n== MODALITY CLASSIFICATION SUMMARY ==")
    
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{k}: {v}")
    
    # Optional: save full results to file
    with open("classified_image_types.tsv", "w") as f:
        f.write("ImagePath\tClassification\n")
        for path, label in image_results.items():
            f.write(f"{path}\t{label}\n")

