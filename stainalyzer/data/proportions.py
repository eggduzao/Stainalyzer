"""
====================================================================================================
proportions.py - placeholder
====================================================================================================

Overview
--------
This module, 'proportions.py', placeholder.


Intended Use Cases
-------------------
- Placeholder.
- Placeholder.

Key Features
-------------
1. **Modularity and Reusability:** The utilities are designed as independent, reusable functions and classes, 
   making them suitable for various projects without modification.
2. **Configurable and Extensible:** The design allows easy extension of functionality through subclassing and 
   dynamic configuration loading.
3. **Performance-Oriented Implementations:** Where performance is critical, optimized algorithms and 
   vectorized operations (e.g., via NumPy) have been employed.
4. **Robust Error Handling:** Defensive programming practices ensure that potential runtime exceptions, 
   such as file I/O errors and data integrity violations, are properly managed and logged.
5. **Integration with External Libraries:** Seamless interoperability with popular libraries like `pandas`, 
   `numpy`, `pyyaml`, and `logging`, ensuring compatibility with established Python ecosystems.

Examples of Usage
-------------------
The following examples illustrate the versatility of the utilities provided within this module:

**Example 1: Placeholder**

```python
Placeholder
```

**Example 2: Placeholder**

```python
Placeholder
```

**Example 3: Placeholder**

```python
Placeholder
```

Development Notes:
-------------

import os
from pathlib import Path
from matplotlib.font_manager import findSystemFonts

for font in findSystemFonts():
    font_path = Path(font)
    if "Consolas" == font_path.stem:
        print(f"Consolas Stem = {font}")

    # import matplotlib.font_manager as fm

    # for font in fm.fontManager.ttflist:
    #     font_name = font.name
    #     if font_name == "Chalkduster":
    #         print(font)

    # from matplotlib.font_manager import FontEntry
    # FontEntry(fname='/Library/Fonts/Consolas.ttf', 
    #           name='Consolas', 
    #           style='normal', 
    #           variant='normal', 
    #           weight=400, 
    #           stretch='normal', 
    #           size='scalable')

    # import matplotlib.pyplot as plt
    # from matplotlib import rcParams

    # rcParams["font.family"] = "Consolas"

 - Python Compatibility: Python 3.10 and above.

 - Required Packages: NumPy, Pandas, logging.
 
 - Testing Framework: The module includes unit tests implemented using pytest to ensure reliability
across different Python environments.
 
 - Code Style Compliance: All code follows PEP 8 guidelines, with additional comments to aid
maintainability for future developers.

Manual:
-------

1. Placeholder.
2. Placeholder.

Usage Example:
--------------

```python
Placeholder
```

Placeholder:
------------------------

```python
Placeholder
```

Placeholder:
----------------------------
1. Placeholder.
2. Placeholder.

Author: Eduardo Gade Gusmao
Created On: 11/12/2024
Last Updated: 15/02/2025
Version: 0.1.0
License: <Currently_Withheld>


# Example Usage TODOOOOOOOOOOOOOOOOOOOOOO
vp = VizPlots()
vp.pie_chart(values=[10, 20, 30], labels=['A', 'B', 'C'])
vp.star_plot({'Metric1': 3, 'Metric2': 5, 'Metric3': 4})
vp.bar_plot([5, 6, 7], x_labels=['X', 'Y', 'Z'])
vp.box_plot([[1, 2, 3], [3, 4, 5]])
vp.distribution_plot([[1, 2, 3, 2, 2], [3, 4, 5, 3, 3]])
vp.polar_plot({'Metric1': 1, 'Metric2': 4, 'Metric3': 3})
"""

###############################################################################
# Imports
###############################################################################

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import f
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import OrderedDict, Counter
from typing import Generator, List, Any, Callable, Dict, Tuple, Optional


###############################################################################
# Constants
###############################################################################

# Constants
SEED = 1987
np.random.seed(SEED)

###############################################################################
# Classes
###############################################################################

class Proportions:
    """
    Placeholder.

    Methods
    -------
    load_proportions(self)
        Placeholder
    """

    def __init__(self):
        """
        B: Nucleus Segmentation
        1. [F1-SEG]  Dice Coefficient (F1-Score for Segmentation)
        2. [IoU]     Intersection over Union (IoU, Jaccard Index)
        3. [AJI]     Aggregated Jaccard Index
        4. [HD95]    Hausdorff Distance (HD95, 95th percentile)
        5. [F1-BDR]  Boundary F1-Score (BF1)
        6. [PQ]      Panoptic Quality (PQ)
        7. [ARI]     Adjusted Rand Index (ARI)
        8. [F1-DET]  Nuclear Detection F1-Score (F1-Detect)
        9. [REC-SEG] Segmentation Recall
        10. [PRE-SEG] Segmentation Precision

        C: High-Resolution Image Enhancement
        1. [PSNR]      Peak Signal-to-Noise Ratio (PSNR)
        2. [SSIM]      Structural Similarity Index (SSIM)
        3. [FSIM]      Feature Similarity Index (FSIM)
        4. [MSE]       Mean Squared Error (MSE)
        5. [NRMSE]     Normalized Root Mean Squared Error (NRMSE)
        6. [GMSD]      Gradient Magnitude Similarity Deviation (GMSD)
        7. [MS-SSIM]   Multi-Scale Structural Similarity Index (MS-SSIM)
        8. [BRISQUE]   Blind/Reference-less Image Spatial Quality Evaluator (BRISQUE)
        9. [NRQM+NIQE] Perceptual Index (PI) = (NRQM + NIQE) / 2
        10. [WB-IQI]   Wavelet-Based Image Quality Index (WB-IQI)

        Image Enhancement:
        1.  Denoising
            Problem: Image is corrupted by noise (Gaussian, Poisson, salt & pepper, etc.)
            Goal: Remove noise while preserving structure.
        2.  Super-resolution
            Problem: Image has low resolution (limited spatial detail).
            Goal: Create a higher-resolution image with plausible added detail.
        3.  Deblurring
            Problem: Image is blurred due to motion or lens issues.
            Goal: Recover sharpness.
        4.  Contrast enhancement
            Problem: Poor dynamic range; hard to see objects in dark or bright areas.
            Goal: Redistribute intensities (e.g., via histogram equalization, CLAHE).
        5.  Illumination correction / Low-light enhancement
            Problem: Image taken in suboptimal lighting.
            Goal: Make it look like it was taken in normal light.
        6.  Color correction / Restoration
            Problem: Faded or distorted colors, often in old photos or underwater images.
            Goal: Restore natural-looking color balance.
        7.  Dehazing / Desmoking / Deraining
            Problem: Atmospheric or environmental interference (fog, smoke, rain).
            Goal: Recover a clean scene.
        8.  Artifact removal
            Problem: Compression artifacts (e.g., JPEG blocks), sensor noise, or stitching problems.
            Goal: Clean up without hallucinating false details.
        9.  Edge or texture enhancement
            Problem: Details are too soft or subtle.
            Goal: Highlight edges or fine structures (often for medical or microscopy).
        10. Multi-modal enhancement
            Problem: Combining multiple types of images (e.g., multi-focus, multi-exposure,
            or even different sensors like MRI + PET).
            Goal: Fuse into a more informative single image.
        """

        # Loading proportions
        self.proportions_dict = OrderedDict()
        self.load_proportions()

        # Metrics
        self.metrics = OrderedDict()
        self.metrics["Nucl"] = ["F1SEG", "IoU", "AJI", "HD95", "F1BDR", "PQ", "ARI", "F1DET", "RECSEG", "PRESEG"]
        self.metrics["Enhc"] = ["PSNR", "SSIM", "FSIM", "MSE", "NRMSE", "GMSD", "MSSSIM", "BRISQUE", "NRQMNIQE", "WBIQI"]


    def load_proportions(self):
        """
        Replaces the old proportions dictionary
        # Creating proportions dictionary with order [1,2,3,4,5]
        proportions_dict = {
            1: [94, 5, 1, 0, 0], 2: [88, 9, 2, 1, 0], 3: [84, 10, 3, 3, 0], 4: [80, 11, 5, 4, 0],
            5: [77, 12, 7, 4, 0], 6: [72, 17, 7, 4, 0], 7: [70, 19, 8, 2, 1], 8: [66, 20, 9, 3, 2],
            9: [60, 22, 11, 4, 3], 10: [56, 23, 11, 7, 3], 11: [53, 23, 13, 7, 4], 12: [52, 22, 13, 8, 5],
            13: [51, 22, 13, 8, 6], 14: [50, 25, 12, 7, 6], 15: [48, 25, 12, 7, 8], 16: [46, 24, 13, 9, 8],
            17: [45, 25, 13, 8, 9], 18: [43, 24, 14, 10, 9], 19: [42, 25, 14, 9, 10], 20: [40, 24, 14, 12, 10],
            21: [37, 23, 19, 10, 11], 22: [36, 23, 19, 11, 11], 23: [35, 23, 21, 9, 12], 24: [34, 21, 23, 10, 12],
            25: [30, 22, 24, 12, 12], 26: [29, 22, 24, 12, 13], 27: [27, 19, 28, 13, 13], 28: [25, 19, 29, 13, 14],
            29: [22, 19, 32, 13, 14], 30: [21, 18, 32, 16, 13], 31: [19, 18, 34, 16, 13], 32: [17, 17, 35, 18, 13],
            33: [16, 17, 36, 18, 13], 34: [15, 16, 36, 19, 14], 35: [11, 15, 38, 20, 16], 36: [10, 15, 41, 17, 17],
            37: [7, 16, 41, 17, 19], 38: [5, 13, 39, 22, 21], 39: [4, 13, 37, 25, 21], 40: [0, 12, 37, 29, 22],
            41: [0, 9, 36, 32, 23], 42: [0, 9, 36, 32, 23], 43: [0, 8, 35, 33, 24], 44: [0, 7, 35, 33, 25],
            45: [0, 6, 33, 35, 26], 46: [0, 6, 30, 38, 26], 47: [0, 5, 27, 42, 26], 48: [0, 4, 26, 42, 28],
            49: [0, 3, 24, 45, 28], 50: [0, 2, 23, 47, 28], 51: [0, 0, 22, 50, 28], 52: [0, 0, 20, 50, 30],
            53: [0, 0, 14, 52, 34],
        }
        """

        # Creating proportions dictionary with order [1,2,3,4,5]
        self.proportions_dict = OrderedDict(
            {
            1: [96, 3, 1, 0, 0],
            2: [93, 4, 2, 1, 0],
            3: [90, 5, 3, 2, 0],
            4: [87, 7, 3, 3, 0],
            5: [84, 7, 4, 4, 1],
            6: [81, 8, 5, 5, 1],
            7: [78, 10, 5, 6, 1],
            8: [75, 11, 6, 7, 1],
            9: [72, 12, 6, 8, 2],
            10: [69, 13, 7, 9, 2],
            11: [66, 14, 8, 10, 2],
            12: [63, 15, 9, 11, 2],
            13: [60, 16, 9, 12, 3],
            14: [57, 18, 9, 13, 3],
            15: [54, 18, 11, 14, 3],
            16: [48, 19, 15, 15, 3],
            17: [45, 19, 16, 16, 4],
            18: [43, 20, 16, 17, 4],
            19: [42, 20, 16, 18, 4],
            20: [40, 21, 17, 18, 4],
            21: [39, 21, 17, 19, 4],
            22: [36, 22, 17, 20, 5],
            23: [35, 22, 18, 20, 5],
            24: [34, 21, 19, 21, 5],
            25: [32, 21, 19, 22, 6],
            26: [30, 20, 19, 24, 7],
            27: [27, 20, 20, 26, 7],
            28: [25, 20, 21, 27, 7],
            29: [24, 19, 21, 28, 8],
            30: [23, 18, 22, 29, 8],
            31: [22, 17, 23, 29, 9],
            32: [21, 16, 24, 30, 9],
            33: [18, 15, 26, 31, 10],
            34: [15, 14, 30, 31, 10],
            35: [12, 13, 32, 32, 11],
            36: [10, 12, 33, 33, 12],
            37: [9, 11, 34, 34, 12],
            38: [6, 11, 34, 35, 14],
            39: [4, 11, 36, 35, 14],
            40: [3, 10, 37, 35, 15],
            41: [2, 9, 38, 36, 15],
            42: [0, 9, 39, 36, 16],
            43: [0, 8, 39, 37, 16],
            44: [0, 7, 38, 38, 17],
            45: [0, 7, 37, 39, 17],
            46: [0, 6, 36, 40, 18],
            47: [0, 6, 35, 41, 18],
            48: [0, 5, 34, 42, 19],
            49: [0, 5, 33, 43, 19],
            50: [0, 5, 32, 44, 19],
            51: [0, 4, 32, 44, 20],
            52: [0, 3, 32, 45, 20],
            53: [0, 3, 32, 45, 20],
            54: [0, 3, 31, 46, 20],
            55: [0, 2, 31, 46, 21],
            56: [0, 2, 30, 47, 21],
            57: [0, 1, 29, 48, 22],
            58: [0, 1, 28, 49, 22],
            59: [0, 0, 25, 52, 23],
            60: [0, 0, 24, 53, 23],
            61: [0, 0, 23, 54, 23],
            62: [0, 0, 22, 54, 24],
            63: [0, 0, 22, 53, 25],
            64: [0, 0, 21, 54, 25],
            65: [0, 0, 20, 54, 26],
            66: [0, 0, 19, 55, 26],
            67: [0, 0, 18, 55, 27],
            68: [0, 0, 17, 56, 27],
            69: [0, 0, 16, 56, 28],
            70: [0, 0, 15, 57, 28],
            71: [0, 0, 13, 58, 29],
            72: [0, 0, 12, 59, 29],
            73: [0, 0, 11, 59, 30],
            74: [0, 0, 10, 60, 30],
            75: [0, 0, 9, 60, 31],
            76: [0, 0, 8, 61, 31],
        })

    def get_discrete_metric_interval(self, minn, maxn, function, *args, **kwargs):
        """
        for i in range(n):\n",
            modified_params = {key: value + i for key, value in kwargs.items()}
            sample = func(size=size, **modified_params)

        Convert string to number and vice-versa:
        num = float("20.5")  # 20.5
        num_str = str(20.5)  # "20.5"
        """
        number = function(*args, **kwargs)
        number = np.clip(number, minn, maxn)
        return number        

    def get_continuous_metric_interval(self, minn, maxn, function, *args, **kwargs):
        """
        get_continuous_metric_interval
        """
        number = function(*args, **kwargs)
        number = np.clip(number, minn, maxn)
        return number

    def get_discrete_metric_open(self, hard_min, hard_max, function, soft_min=1, soft_max=98, *args, **kwargs):
        """
        get_discrete_metric_open
        """
        number = function(*args, **kwargs)
        number = np.clip(number, hard_min, hard_max)
        soft_sub = int(self._get_glitch(maxN=3, mult=15))
        soft_add = int(self._get_glitch(maxN=3, mult=15))
        number = np.clip(number, soft_min - soft_sub, soft_max + soft_add)
        return number 

    def get_continuous_metric_open(self, hard_min, hard_max, function, soft_min=0.1, soft_max=0.98, *args, **kwargs):
        """
        get_continuous_metric_open
        """
        number = function(*args, **kwargs)
        number = np.clip(number, hard_min, hard_max)
        soft_sub = int(self._get_glitch(maxN=6, mult=10))
        soft_add = int(self._get_glitch(maxN=6, mult=10))
        number = np.clip(number, soft_min - soft_sub, soft_max + soft_add)
        return number

    def _string_to_unicode_sum(self, word):
        return sum(ord(char) for char in word)

    def sum_digits(self, number):
        return sum(int(digit) for digit in str(abs(number)) if digit.isdigit())

    def _get_glitch(self, maxN=4, mult=10):
        glitch_list = []
        random_digit = np.random.randint(0, 10)
        for i in range(1, 10**maxN, 10):
            glitch_n = (random_digit * mult) / (i * 10)
            glitch_list.append(glitch_n)
        return np.random.choice()

    def create_nucleus_tables(self):
        """
        Database Names: [NB]_[DATABASE]_Nucl_[METRIC]
        Database Shapes: Images x Methods
        ----------------------------------------------------
        Example: 100_SegPath_Nucl_F1SEG.tsv
        Image        Method1    Method2     ...     MethodN
        Image000000  0.85       0.82        ...     0.95
        Image000001  0.84       0.57        ...     0.74
        Image000002  0.57       0.62        ...     0.86
        ...          ...        ...         ...     ...
        Image292398  0.77       0.71        ...     0.69
        ----------------------------------------------------
        Numbers: [100, 209]
        """
        pass

    def create_enhancer_tables(self):
        """
        Database Names: [NB]_[DATABASE]_Enhc_[METRIC]
        Database Shapes: Images x Methods
        ----------------------------------------------------
        Example: 210_SegPath_Enhc_PSNR.tsv
        Image        Method1    Method2     ...     MethodN
        Image000000  0.85       0.82        ...     0.95
        Image000001  0.84       0.57        ...     0.74
        Image000002  0.57       0.62        ...     0.86
        ...          ...        ...         ...     ...
        Image292398  0.77       0.71        ...     0.69
        ----------------------------------------------------
        Numbers: [210, 319]
        """
        pass
