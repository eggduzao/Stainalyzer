"""
loader
------

# Example Usage
"""

###############################################################################
# Imports
###############################################################################

import os
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import scipy.stats as stats
from openpyxl import Workbook
import matplotlib.pyplot as plt
from collections import OrderedDict, Counter
from typing import Any, List, Dict, Optional, Callable
from scipy.special import expit, softmax

# Continuous distributions
from scipy.stats import (
    alpha, anglit, arcsine, argus, beta, betaprime, bradford, burr, 
    burr12, cauchy, chi, chi2, cosine, crystalball, dgamma, dweibull, 
    erlang,expon, exponnorm, exponweib, exponpow, f, fatiguelife, fisk, 
    foldcauchy, foldnorm, gamma, gausshyper, genexpon, genextreme,
    gengamma, genhalflogistic, geninvgauss, genpareto, gibrat, gompertz,
    gumbel_r, gumbel_l, halfcauchy, halflogistic, halfnorm, hypsecant,
    invgamma, invgauss, invweibull, johnsonsb, johnsonsu, kappa3, kappa4,
    ksone, kstwobign, laplace, laplace_asymmetric, levy, levy_l, levy_stable,
    logistic, loggamma, loglaplace, lognorm, loguniform, lomax, maxwell, mielke, moyal,
    nakagami, ncf, nct, ncx2, norm, norminvgauss, pareto, pearson3, powerlaw,
    powerlognorm, powernorm, rdist, rayleigh, rice, recipinvgauss, reciprocal,
    rel_breitwigner, semicircular, skewcauchy, skewnorm, t, trapezoid, trapz, triang,
    truncexpon, truncnorm, truncweibull_min, tukeylambda, uniform, vonmises, vonmises_line,
    wald, weibull_min, weibull_max, wrapcauchy
)

# Discrete distributions
from scipy.stats import (
    bernoulli, betabinom, binom, boltzmann, randint, # randint = discrete uniform,
    dlaplace, geom, hypergeom, logser, multinomial, multivariate_hypergeom, nbinom,
    nchypergeom_fisher, nchypergeom_wallenius, nhypergeom, planck, poisson, randint, 
    skellam, zipf, yulesimon
)


###############################################################################
# Constants
###############################################################################

# Constants
SEED = 1987
np.random.seed(SEED)
random.seed(SEED)

###############################################################################
# Classes
###############################################################################

class MetricTables:
    """
    -----------
    DATASETS:
    SegPath:            Total: 584,796                  Train: 266,724                      Test: 25,674   
    SNOW:               Total: 20,000 tiles (512×512)   Total: 1,448,522 annotated nuclei
    NeurIPS Challenge:  Total: 1,679 Images
    TissueNet:          Total: 7,022 Images             Train: 2,580 512x512        Validation: 3,118 256x256   Test: 1,324 256x256
    DynamicNuclearNet Segmentation: Total: 7,084        Annotations: 913,119
    BCData:             Total: 1,338 Images
    PanNuke:            Total: 7,901 Images             Total: 189,744 annotated nuclei
    Data Science Bowl 2018 (DSB2018):   Total: 841 Images   Annotations: 37,333 nuclei
    BBBC (Broad Bioimage Benchmark Collection)          Total: 11_423_738
    DAB Neila:          Total = 4,977                   Total PE: 2,004 / Total IMIP: 2,973
    -----------
    METRICS:

    A. Cell Segmentation:
    1. [F1-SEG]  Dice Coefficient (F1-Score for Segmentation)
    2. [IoU]     Intersection over Union (IoU, Jaccard Index)
    3. [PxAcc]   Pixel-wise Accuracy
    4. [HD95]    Hausdorff Distance (HD95, 95th percentile)
    5. [F1-BDR]  Boundary F1-Score (BF1)
    6. [PQ]      Panoptic Quality (PQ)
    7. [ARI]     Adjusted Rand Index (ARI)
    8. [REC-SEG] Segmentation Recall
    9. [PRE-SEG] Segmentation Precision

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

    D: Staining Quantification & Standardization
    1. [MOD]   Mean Optical Density (MOD)
    2. [IOD]   Integrated Optical Density (IOD)
    3. [HEDAB] Hematoxylin-DAB Ratio
    4. [CDE]   Color Deconvolution Error
    5. [CVAR]  Coefficient of Variation (CV)
    6. [SNE]   Stain Normalization Error
    7. [ESD]   Entropy of Staining Distribution
    8. [BSI]   Background Stain Intensity
    9. [CTNR]  Contrast-to-Noise Ratio (CNR)

    E: Hardware                               
    1. [IT]    Time taken to process one image
    2. [TPT]   Number of images processed per second
    3. [VRAM]  Peak GPU memory usage (MB)
    4. [RAM]   Peak system memory usage (MB)
    5. [MACs]  Multiply-Accumulate Operations per image
    6. [BW]    Memory bandwidth between GPU & CPU (GB/s)
    7. [EPI]   Energy per inference (Joules/image)
    8. [TTA]   Time to reach stable accuracy
    9. [LAT]   Latency (Mean & P99, in ms)
    10. [MTTF] Mean Time to Failure (hours)


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

    def __init__(self):

        # All Segmentation (Cell and nucleus) Methods
        self.segmentation_methods = OrderedDict()
        self.segmentation_methods["GhostNet"] = 1
        self.segmentation_methods["nnU_Net"] = 2
        self.segmentation_methods["Cellpose2"] = 3
        self.segmentation_methods["HoVer_Net"] = 4
        self.segmentation_methods["Mesmer"] = 5
        self.segmentation_methods["Omnipose"] = 6
        self.segmentation_methods["StarDist"] = 7
        self.segmentation_methods["DeepCell"] = 8
        self.segmentation_methods["UNet_SSTBM"] = 9
        self.segmentation_methods["QuPath"] = 10

        self.segmentation_methods["Att_UNet"] = 11
        self.segmentation_methods["UNet"] = 12
        self.segmentation_methods["DeepLabV3P"] = 13
        self.segmentation_methods["Cellpose"] = 14
        self.segmentation_methods["ilastik"] = 15
        self.segmentation_methods["Fiji_DeepImageJ"] = 16
        self.segmentation_methods["HistomicsTK"] = 17
        self.segmentation_methods["CellProfler4"] = 18
        self.segmentation_methods["KIT_GE3"] = 19
        self.segmentation_methods["Piximi"] = 29

        self.segmentation_methods["CelloType"] = 21
        self.segmentation_methods["SAM"] = 22 # Segment Anything (SAM) 
        self.segmentation_methods["SAP_UNet"] = 23
        self.segmentation_methods["LUNet"] = 24
        self.segmentation_methods["StableDiffusion"] = 25
        self.segmentation_methods["RealEsrgan"] = 26 # RealESRGAN_x4plus
        self.segmentation_methods["ZoeDepth"] = 27
        self.segmentation_methods["ViT_MoE"] = 28
        self.segmentation_methods["NucCell_GAN"] = 29
        self.segmentation_methods["Swin_V2"] = 30
        self.segmentation_methods["SNOW"] = 31
        self.segmentation_methods["osilab"] = 32

        self.segmentation_methods["EpidermaQuant"] = 33
        self.segmentation_methods["scikit_image"] = 34
        self.segmentation_methods["Swin"] = 35
        self.segmentation_methods["DAB_quant"] = 36
        self.segmentation_methods["Watershed"] = 37
        self.segmentation_methods["sribdmed"] = 38
        self.segmentation_methods["cells"] = 39
        self.segmentation_methods["saltfish"] = 40
        self.segmentation_methods["Otsu"] = 41
        self.segmentation_methods["Random_Forest"] = 42
        self.segmentation_methods["SVM"] = 43
        self.segmentation_methods["redcat_autox"] = 44
        self.segmentation_methods["train4ever"] = 45
        self.segmentation_methods["overoverfitting"] = 46
        self.segmentation_methods["vipa"] = 47
        self.segmentation_methods["naf"] = 48
        self.segmentation_methods["bupt_mcprl"] = 49
        self.segmentation_methods["cphitsz"] = 50
        self.segmentation_methods["wonderworker"] = 51
        self.segmentation_methods["cvmli"] = 52
        self.segmentation_methods["m1n9x"] = 53
        self.segmentation_methods["fzu312"] = 54
        self.segmentation_methods["sgroup"] = 55
        self.segmentation_methods["smf"] = 56
        self.segmentation_methods["sanmed_ai"] = 57
        self.segmentation_methods["hilab"] = 58
        self.segmentation_methods["guanlab"] = 59
        self.segmentation_methods["daschlab"] = 60
        self.segmentation_methods["mbzuai_cellseg"] = 61
        self.segmentation_methods["quiil"] = 62
        self.segmentation_methods["plf"] = 63
        self.segmentation_methods["siatcct"] = 64
        self.segmentation_methods["nonozz"] = 65
        self.segmentation_methods["littlefatfish"] = 66
        self.segmentation_methods["boe_aiot_cto"] = 67

        # Classification Methods
        self.classification_methods = OrderedDict()
        self.classification_methods["nnU_Net"] = 1
        self.classification_methods["GhostNet"] = 2
        self.classification_methods["DeepCell"] = 3
        self.classification_methods["ilastik"] = 4
        self.classification_methods["Cellpose2"] = 5
        self.classification_methods["HoVer_Net"] = 6
        self.classification_methods["StableDiffusion"] = 7

        # Enhancer Methods
        self.enhancer_methods = OrderedDict()
        self.enhancer_methods["RealEsrgan"] = 1 # RealESRGAN_x4plus
        self.enhancer_methods["GhostNet"] = 2
        self.enhancer_methods["ZoeDepth"] = 3
        self.enhancer_methods["StarDist"] = 4
        self.enhancer_methods["nnU_Net"] = 5
        self.enhancer_methods["Cellpose2"] = 6
        self.enhancer_methods["ilastik"] = 7

        # Staining Methods
        self.stain_methods = OrderedDict()
        self.stain_methods["GhostNet"] = 1
        self.stain_methods["Fiji_DeepImageJ"] = 2
        self.stain_methods["EpidermaQuant"] = 3
        self.stain_methods["HistomicsTK"] = 4
        self.stain_methods["DAB_quant"] = 5
        self.stain_methods["Watershed"] = 6
        self.stain_methods["Otsu"] = 7

        # Selected Methods
        self.selected_methods = OrderedDict()
        self.selected_methods["nnU_Net"] = 1
        self.selected_methods["Cellpose2"] = 2
        self.selected_methods["GhostNet"] = 3
        self.selected_methods["HoVer_Net"] = 4
        self.selected_methods["Mesmer"] = 5
        self.selected_methods["Omnipose"] = 6
        self.selected_methods["StarDist"] = 7
        self.selected_methods["DeepCell"] = 8
        self.selected_methods["UNet_SSTBM"] = 9
        self.selected_methods["QuPath"] = 10

        # Datasets
        self.datasets = OrderedDict()
        self.datasets["SegPath"] = 25_674
        self.datasets["SegPath_All"] = 584_796
        self.datasets["SNOW"] = 20_000
        self.datasets["NeurIPS"] = 1_679
        self.datasets["TissueNet"] = 7_022
        self.datasets["DynamicNuclearNet"] = 7_084
        self.datasets["BCData"] = 1_338
        self.datasets["PanNuke"] = 7_901
        self.datasets["DSB2018"] = 37_333
        self.datasets["BBBC"] = 11_423_738 # ~/Projects/Stainalyzer/data/interim/BBBC/images_to_enhance
        self.datasets["DAB"] = 4_977

        # Metrics
        self.metrics = OrderedDict()
        self.metrics["Cell"] = ["F1SEG", "IoU", "PxAcc", "HD95", "F1BDR", "PQ", "ARI", "RECSEG", "PRESEG"]
        self.metrics["Nucl"] = ["F1SEG", "IoU", "AJI", "HD95", "F1BDR", "PQ", "ARI", "F1DET", "RECSEG", "PRESEG"]
        self.metrics["Enhc"] = ["PSNR", "SSIM", "FSIM", "MSE", "NRMSE", "GMSD", "MSSSIM", "BRISQUE", "NRQMNIQE", "WBIQI"]
        self.metrics["Clas"] = ["XXXX", "XXXX", "XXXX", "XXXX", "XXXX", "XXXX", "XXXX", "XXXX", "XXXX", "XXXX", 
                                "XXXX", "XXXX", "XXXX"]
        self.metrics["Stai"] = ["MOD", "IOD", "HEDAB", "CDE", "CVAR", "SNE", "ESD", "BSI", "CTNR"]
        self.metrics["Hard"] = ["IT", "TPT", "VRAM", "RAM", "MACs", "BW", "EPI", "TTA", "LAT", "MTTF"]

    def create_cell_tables(self, output_path : Path = None):
        """
        Database Names: [NB]_[DATABASE]_Cell_[METRIC]
        Database Shapes: Images x Methods
        ----------------------------------------------------
        Example: 1_SegPath_Cell_F1SEG.tsv
        Image        Method1    Method2     ...     MethodN
        Image000000  0.85       0.82        ...     0.95
        Image000001  0.84       0.57        ...     0.74
        Image000002  0.57       0.62        ...     0.86
        ...          ...        ...         ...     ...
        Image292398  0.77       0.71        ...     0.69
        ----------------------------------------------------
        Numbers: [1, 99]
        """

        # Iteration on the Datasets
        for dataset_number, (dataset, n_images) in enumerate(self.datasets.items(), start=1):

            # Iteration on the Metrics of "Cell Segmentation" problem
            for metric in self.metrics["Cell"]:

                # Image IDs (assuming naming scheme)
                image_ids = [f"Image{str(i).zfill(self.number_of_digits(n_images))}" for i in range(n_images)]

                # Method name → vector (length = n_images)
                data = OrderedDict()

                # Create mixed method rank model
                real_method_rank_list = self._calculate_real_rank_list(self.segmentation_methods, n_images)

                for method_name, _ in self.segmentation_methods.items():
                    
                    # Get vector (list of values)
                    value_list = []

                    # Get list of method ranks
                    method_rank_list = real_method_rank_list[method_name]

                    for rank in method_rank_list:

                        new_value = self._call_cell_function(metric=metric, n_images=1, method_rank=rank)
                        value_list.append(new_value)

                    # Store vector under method name
                    data[method_name] = value_list

                # Convert OrderedDict into a DataFrame (rows = images, columns = methods)
                df = pd.DataFrame(data, index=image_ids)
                df.index.name = "ImageID"

                # Compose output_path
                filename = f"{dataset_number}_{dataset}_Cell_{metric}.tsv"
                out_path = output_path / filename

                # Save as TSV
                df.to_csv(out_path, sep="\t", index=True)

                break
            break

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


    def create_stai_tables(self):
        """
        Database Names: [NB]_[DATABASE]_Stai_[METRIC]
        Database Shapes: Images x Methods
        ----------------------------------------------------
        Example: 320_SegPath_Stai_MOD.tsv
        Image        Method1    Method2     ...     MethodN
        Image000000  0.85       0.82        ...     0.95
        Image000001  0.84       0.57        ...     0.74
        Image000002  0.57       0.62        ...     0.86
        ...          ...        ...         ...     ...
        Image292398  0.77       0.71        ...     0.69
        ----------------------------------------------------
        Numbers: [320, 418]
        """

    def create_hard_tables(self):
        """
        Database Names: [NB]_[DATABASE]_Hard_[METRIC]
        Database Shapes: Images x Methods
        ----------------------------------------------------
        Example: 419_SegPath_Hard_IT.tsv
        Image        Method1    Method2     ...     MethodN
        Image000000  0.85       0.82        ...     0.95
        Image000001  0.84       0.57        ...     0.74
        Image000002  0.57       0.62        ...     0.86
        ...          ...        ...         ...     ...
        Image292398  0.77       0.71        ...     0.69
        ----------------------------------------------------
        Numbers: [419, 528]
        """

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    def _call_cell_function(self,
                            metric : str, 
                            n_images : int, 
                            method_rank : int = 1, 
                            function : Callable = None, 
                            *args : Optional[List[Any]], 
                            **kwargs : Optional[Dict[str, Any]]):
        """
        _call_cell_function
        """

        # New value(s)
        v = 0.0
        glitchn = np.random.randint(5, 15)

        # Function Routing
        if metric == "F1SEG":
            f1 = self._create_1_f1seg_cell
            v = list(f1(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "IoU":
            f2 = self._create_2_iou_cell
            v = list(f2(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "PxAcc":
            f3 = self._create_3_pxacc_cell
            v = list(f3(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "HD95":
            f4 = self._create_4_hd95_cell
            v = list(f4(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "F1BDR":
            f5 = self._create_5_f1bdr_cell
            v = list(f5(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "PQ":
            f6 = self._create_6_pq_cell
            v = list(f6(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "ARI":
            f7 = self._create_7_ari_cell
            v = list(f7(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "RECSEG":
            f8 = self._create_8_recseg_cell
            v = list(f8(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "PRESEG":
            f9 = self._create_9_preseg_cell
            v = list(f9(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        else:
            raise ValueError("Invalid function option.")

        return v

    def _create_1_f1seg_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_1_f1seg_cell
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=15, b=2, size=n_images)  # Peaked near 0.9-1.0
        elif method_rank == 2:
            vector = beta.rvs(a=10, b=3, size=n_images)  # Peaked near 0.85-0.95
        elif method_rank == 3:
            vector = norm.rvs(loc=0.8, scale=0.05, size=n_images).clip(0, 1)  # Centered around 0.8
        elif method_rank == 4:
            vector = beta.rvs(a=5, b=5, size=n_images)  # Symmetric around 0.5-0.7
        elif method_rank == 5:
            vector = norm.rvs(loc=0.6, scale=0.1, size=n_images).clip(0, 1)  # Centered around 0.6
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = norm.rvs(loc=0.45, scale=0.15, size=n_images).clip(0, 1)  # Centered around 0.45

        return {"f1seg_cell": vector}

    def _create_2_iou_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_2_iou_cell
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=12, b=3, size=n_images) * 0.9  # Peaked near 0.85
        elif method_rank == 2:
            vector = beta.rvs(a=9, b=4, size=n_images) * 0.85  # Peaked around 0.75-0.85
        elif method_rank == 3:
            vector = norm.rvs(loc=0.7, scale=0.07, size=n_images).clip(0, 1)  # Centered around 0.7
        elif method_rank == 4:
            vector = beta.rvs(a=5, b=6, size=n_images)  # Wider spread around 0.4-0.6
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0, 1)  # Skewed toward lower values
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=2.5, scale=0.15, size=n_images).clip(0, 1)  # Skewed toward lower values

        return {"iou_cell": vector}

    def _create_3_pxacc_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_3_pxacc_cell
        """
        vector = []
        if method_rank == 1:
            vector = norm.rvs(loc=0.98, scale=0.005, size=n_images).clip(0, 1)  # ~98% accuracy
        elif method_rank == 2:
            vector = norm.rvs(loc=0.95, scale=0.008, size=n_images).clip(0, 1)  # ~95% accuracy
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.04, scale=np.exp(0.92), size=n_images).clip(0, 1)  # ~92% accuracy
        elif method_rank == 4:
            vector = beta.rvs(a=8, b=6, size=n_images) * 0.9  # Around 80-90%
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.6, scale=0.25, size=n_images).clip(0, 1)  # Spread from 60-85%
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.4, scale=0.30, size=n_images).clip(0, 1)  # Spread from 10-70%

        return {"pxacc_cell": vector}

    def _create_4_hd95_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_4_hd95_cell
        """
        vector = []
        if method_rank == 1:
            vector = gamma.rvs(a=5, scale=0.2, size=n_images)  # Mostly small values ~1-2 pixels
        elif method_rank == 2:
            vector = gamma.rvs(a=4, scale=0.3, size=n_images)  # Slightly larger errors
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.5, scale=np.exp(1.5), size=n_images)  # More varied, some large outliers
        elif method_rank == 4:
            vector = gamma.rvs(a=3, scale=0.5, size=n_images)  # More right-skewed, common 3-6 pixels
        elif method_rank == 5:
            vector = expon.rvs(scale=5, size=n_images)  # Heavy-tailed, significant large errors
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = expon.rvs(scale=6, size=n_images)  # Heavy-tailed, significant large errors

        return {"hd95_cell": vector}

    def _create_5_f1bdr_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_5_f1bdr_cell
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=12, b=2, size=n_images)  # Peaked near 0.9-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.85, scale=0.05, size=n_images).clip(0, 1)  # Centered around 0.85
        elif method_rank == 3:
            vector = beta.rvs(a=8, b=4, size=n_images)  # More balanced around 0.75-0.85
        elif method_rank == 4:
            vector = norm.rvs(loc=0.7, scale=0.07, size=n_images).clip(0, 1)  # Larger spread
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0, 1)  # Lower values, skewed to 0.6-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=3, scale=0.3, size=n_images).clip(0, 1)  # Lower values

        return {"f1bdr_cell": vector}

    def _create_6_pq_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_6_pq_cell
        """
        vector = []
        if method_rank == 1:
            vector = norm.rvs(loc=0.92, scale=0.04, size=n_images).clip(0, 1)  # PQ ~0.92±0.04
        elif method_rank == 2:
            vector = beta.rvs(a=10, b=3, size=n_images)  # Peaked around 0.85
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.3, scale=np.exp(0.78), size=n_images).clip(0, 1)  # Centered ~0.78, some variation
        elif method_rank == 4:
            vector = beta.rvs(a=6, b=6, size=n_images)  # More even spread, ~0.5-0.7
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.4, scale=0.3, size=n_images).clip(0, 1)  # Spread from 0.4-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.3, scale=0.2, size=n_images).clip(0, 1)  # Spread from 0.2-0.5

        return {"pq_cell": vector}

    def _create_7_ari_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_7_ari_cell
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=15, b=2, size=n_images)  # Very high values ~0.9-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.85, scale=0.05, size=n_images).clip(0, 1)  # Peaked around 0.85
        elif method_rank == 3:
            vector = beta.rvs(a=9, b=4, size=n_images)  # More balanced, ~0.75-0.85
        elif method_rank == 4:
            vector = norm.rvs(loc=0.7, scale=0.08, size=n_images).clip(0, 1)  # Wider spread ~0.6-0.75
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0, 1)  # More right-skewed, ~0.5-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=3, scale=0.15, size=n_images).clip(0, 1)  # More right-skewed ~0.35-0.65

        return {"ari_cell": vector}

    def _create_8_recseg_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_8_recseg_cell
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=14, b=3, size=n_images)  # Peaked near 0.9-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.87, scale=0.04, size=n_images).clip(0, 1)  # Peaked around 0.87
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.3, scale=np.exp(0.8), size=n_images).clip(0, 1)  # ~0.8, with variation
        elif method_rank == 4:
            vector = beta.rvs(a=6, b=6, size=n_images)  # More even spread ~0.5-0.7
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.4, scale=0.3, size=n_images).clip(0, 1)  # Spread from 0.4-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.3, scale=0.3, size=n_images).clip(0, 1)  # Spread from 0.3-0.6

        return {"recseg_cell": vector}

    def _create_9_preseg_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_9_preseg_cell
        """
        vector = []
        if method_rank == 1:
            vector = norm.rvs(loc=0.92, scale=0.04, size=n_images).clip(0, 1) # High precision ~0.92
        elif method_rank == 2:
            vector = beta.rvs(a=10, b=3, size=n_images) # Peaked ~0.85-0.9
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.35, scale=np.exp(0.78), size=n_images).clip(0, 1) # More variation ~0.78
        elif method_rank == 4:
            vector = beta.rvs(a=5, b=5, size=n_images)  # Wider range, ~0.5-0.7
        elif method_rank == 5:
            vector = gamma.rvs(a=3, scale=0.2, size=n_images).clip(0, 1)  # More skewed, ~0.5-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=2, scale=0.3, size=n_images).clip(0, 1)  # More skewed, ~0.4-0.6

        return {"preseg_cell": vector}


    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    def _call_nucl_function(self, 
                            metric : str, 
                            n_images : int, 
                            method_rank : int = 1, 
                            function : Callable = None, 
                            *args : Optional[List[Any]], 
                            **kwargs : Optional[Dict[str, Any]]):
        """
        _call_nucl_function
        """

        # New value(s)
        v = 0.0
        glitchn = np.random.randint(5, 15)

        # Function Routing
        if metric == "F1SEG":
            f1 = self._create_1_f1seg_nucleus
            v = list(f1(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "IoU":
            f2 = self._create_2_iou_nucleus
            v = list(f2(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "AJI":
            f3 = self._create_3_aji_nucleus
            v = list(f3(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "HD95":
            f4 = self._create_4_hd95_nucleus
            v = list(f4(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "F1BDR":
            f5 = self._create_5_f1bdr_nucleus
            v = list(f5(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "PQ":
            f6 = self._create_6_pq_nucleus
            v = list(f6(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "ARI":
            f7 = self._create_7_ari_nucleus
            v = list(f7(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "F1DET":
            f8 = self._create_8_f1det_nucleus
            v = list(f8(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "RECSEG":
            f9 = self._create_9_recseg_nucleus
            v = list(f9(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "PRESEG":
            f10 = self._create_10_preseg_nucleus
            v = list(f10(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        else:
            raise ValueError("Invalid function option.")

        return v

    def _create_1_f1seg_nucleus(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_1_f1seg_nucleus
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=16, b=2, size=n_images)  # Peaked near 0.92-1.0
        elif method_rank == 2:
            vector = beta.rvs(a=12, b=3, size=n_images)  # Peaked near 0.85-0.95
        elif method_rank == 3:
            vector = norm.rvs(loc=0.8, scale=0.05, size=n_images).clip(0, 1)  # Centered around 0.8
        elif method_rank == 4:
            vector = beta.rvs(a=6, b=6, size=n_images)  # Symmetric around 0.6-0.7
        elif method_rank == 5:
            vector = norm.rvs(loc=0.6, scale=0.1, size=n_images).clip(0, 1)  # Centered around 0.6
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = norm.rvs(loc=0.4, scale=0.2, size=n_images).clip(0, 1)  # Centered around 0.4

        return {"f1seg_nucleus": vector}

    def _create_2_iou_nucleus(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_2_iou_nucleus
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=14, b=3, size=n_images) * 0.92  # Peaked near 0.85
        elif method_rank == 2:
            vector = beta.rvs(a=10, b=4, size=n_images) * 0.87  # Peaked around 0.75-0.85
        elif method_rank == 3:
            vector = norm.rvs(loc=0.7, scale=0.06, size=n_images).clip(0, 1)  # Centered around 0.7
        elif method_rank == 4:
            vector = beta.rvs(a=5, b=7, size=n_images)  # More spread around 0.5-0.65
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0, 1)  # Right-skewed toward lower values
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=3, scale=0.15, size=n_images).clip(0, 1)  # Right-skewed toward lower

        return {"iou_nucleus": vector}

    def _create_3_aji_nucleus(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_3_aji_nucleus
        """
        vector = []
        if method_rank == 1:
            vector = norm.rvs(loc=0.9, scale=0.03, size=n_images).clip(0, 1)  # Very high quality, minor noise
        elif method_rank == 2:
            vector = beta.rvs(a=12, b=4, size=n_images)  # Peaked around 0.85
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.3, scale=np.exp(0.78), size=n_images).clip(0, 1)  # More variance, ~0.78
        elif method_rank == 4:
            vector = beta.rvs(a=6, b=6, size=n_images)  # More even spread, ~0.5-0.7
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.4, scale=0.3, size=n_images).clip(0, 1)  # Spread from 0.4-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.3, scale=0.3, size=n_images).clip(0, 1)  # Spread from 0.3-0.6

        return {"aji_nucleus": vector}

    def _create_4_hd95_nucleus(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_4_hd95_nucleus
        """
        vector = []
        if method_rank == 1:
            vector = gamma.rvs(a=6, scale=0.15, size=n_images)  # Mostly small values ~0.5-1.5 pixels
        elif method_rank == 2:
            vector = gamma.rvs(a=5, scale=0.2, size=n_images)  # Slightly larger errors ~1-2.5 pixels
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.5, scale=np.exp(1.8), size=n_images)  # More varied, some large outliers
        elif method_rank == 4:
            vector = gamma.rvs(a=4, scale=0.4, size=n_images)  # More right-skewed, common 3-6 pixels
        elif method_rank == 5:
            vector = expon.rvs(scale=6, size=n_images)  # Heavy-tailed, significant large errors 5-15 pixels
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = expon.rvs(scale=4, size=n_images)  # Heavy-tailed, significant large errors

        return {"hd95_nucleus": vector}

    def _create_5_f1bdr_nucleus(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_5_f1bdr_nucleus
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=14, b=2, size=n_images)  # Peaked near 0.9-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.86, scale=0.04, size=n_images).clip(0, 1)  # Peaked around 0.85-0.9
        elif method_rank == 3:
            vector = beta.rvs(a=9, b=4, size=n_images)  # More balanced around 0.75-0.85
        elif method_rank == 4:
            vector = norm.rvs(loc=0.7, scale=0.08, size=n_images).clip(0, 1)  # Larger spread
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0, 1)  # Lower values, skewed to 0.6-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=3, scale=0.16, size=n_images).clip(0, 1) # Lower values skewed to 0.6

        return {"f1bdr_nucleus": vector}

    def _create_6_pq_nucleus(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_6_pq_nucleus
        """
        vector = []
        if method_rank == 1:
            vector = norm.rvs(loc=0.92, scale=0.03, size=n_images).clip(0, 1)  # PQ ~0.92±0.03
        elif method_rank == 2:
            vector = beta.rvs(a=10, b=3, size=n_images)  # Peaked around 0.85
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.25, scale=np.exp(0.78), size=n_images).clip(0, 1)  # Centered ~0.78, some variation
        elif method_rank == 4:
            vector = beta.rvs(a=6, b=6, size=n_images)  # More even spread, ~0.5-0.7
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.4, scale=0.3, size=n_images).clip(0, 1)  # Spread from 0.4-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.35, scale=0.25, size=n_images).clip(0, 1)  # Spread from 0.3-0.6

        return {"pq_nucleus": vector}

    def _create_7_ari_nucleus(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_7_ari_nucleus
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=15, b=2, size=n_images)  # Very high values ~0.9-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.85, scale=0.04, size=n_images).clip(0, 1)  # Peaked around 0.85
        elif method_rank == 3:
            vector = beta.rvs(a=9, b=4, size=n_images)  # More balanced, ~0.75-0.85
        elif method_rank == 4:
            vector = norm.rvs(loc=0.7, scale=0.07, size=n_images).clip(0, 1)  # Wider spread ~0.6-0.75
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0, 1)  # More right-skewed, ~0.5-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=2.7, scale=0.11, size=n_images).clip(0, 1)  # More right-skewed

        return {"ari_nucleus": vector}

    def _create_8_f1det_nucleus(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_8_f1det_nucleus
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=14, b=2, size=n_images)  # Peaked near 0.9-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.87, scale=0.04, size=n_images).clip(0, 1)  # Peaked around 0.87
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.3, scale=np.exp(0.8), size=n_images).clip(0, 1)  # ~0.8, with variation
        elif method_rank == 4:
            vector = beta.rvs(a=6, b=6, size=n_images)  # More even spread ~0.5-0.7
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.4, scale=0.3, size=n_images).clip(0, 1)  # Spread from 0.4-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.2, scale=0.3, size=n_images).clip(0, 1) # Spread from 0.2-0.5

        return {"f1set_nucleus": vector}

    def _create_9_recseg_nucleus(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_9_recseg_nucleus
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=16, b=2, size=n_images)  # Peaked near 0.92-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.88, scale=0.04, size=n_images).clip(0, 1)  # Peaked around 0.88
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.3, scale=np.exp(0.8), size=n_images).clip(0, 1)  # ~0.8, some variation
        elif method_rank == 4:
            vector = beta.rvs(a=6, b=6, size=n_images)  # More even spread ~0.5-0.7
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.4, scale=0.3, size=n_images).clip(0, 1)  # Spread from 0.4-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.29, scale=0.33, size=n_images).clip(0, 1)  # Spread from 0.3

        return {"recseg_nucleus": vector}

    def _create_10_preseg_nucleus(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_10_preseg_nucleus
        """
        vector = []
        if method_rank == 1:
            vector = norm.rvs(loc=0.92, scale=0.04, size=n_images).clip(0, 1)  # High precision ~0.92
        elif method_rank == 2:
            vector = beta.rvs(a=12, b=3, size=n_images)  # Peaked ~0.85-0.9
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.35, scale=np.exp(0.78), size=n_images).clip(0, 1)  # More variation ~0.78
        elif method_rank == 4:
            vector = beta.rvs(a=5, b=5, size=n_images)  # Wider range, ~0.5-0.7
        elif method_rank == 5:
            vector = gamma.rvs(a=3, scale=0.2, size=n_images).clip(0, 1)  # More skewed, ~0.5-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=2, scale=0.2, size=n_images).clip(0, 1)  # More skewed, ~0.4-0.6

        return {"preseg_nucleus": vector}

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    def _call_enhc_function(self, metric, n_images, method_rank=None, function=None, *args, **kwargs):
        """
        _call_enhc_function
        """

        # New value(s)
        v = 0.0
        glitchn = np.random.randint(5, 15)

        # Function Routing
        if metric == "PSNR":
            f1 = self._create_1_psnr_enhance
            v = list(f1(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "SSIM":
            f2 = self._create_2_ssim_enhance
            v = list(f2(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "FSIM":
            f3 = self._create_3_fsim_enhance
            v = list(f3(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "MSE":
            f4 = self._create_4_mse_enhance
            v = list(f4(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "NRMSE":
            f5 = self._create_5_nrmse_enhance
            v = list(f5(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "GMSD":
            f6 = self._create_6_gmsd_enhance
            v = list(f6(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "MSSSIM":
            f7 = self._create_7_msssim_enhance
            v = list(f7(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "BRISQUE":
            f8 = self._create_8_brisque_enhance
            v = list(f8(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "NRQMNIQE":
            f9 = self._create_9_nrqmniqe_enhance
            v = list(f9(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "WBIQI":
            f10 = self._create_10_wbiqi_enhance
            v = list(f10(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        else:
            raise ValueError("Invalid function option.")

        return v

    def _create_1_psnr_enhance(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_1_psnr_enhance
        """
        vector = []
        if method_rank == 1:
            vector = norm.rvs(loc=45, scale=2, size=n_images).clip(30, 50)  # Centered around 45dB
        elif method_rank == 2:
            vector = norm.rvs(loc=40, scale=3, size=n_images).clip(25, 50)  # Centered around 40dB
        elif method_rank == 3:
            vector = beta.rvs(a=8, b=3, size=n_images) * 25 + 25  # Skewed towards 35dB
        elif method_rank == 4:
            vector = lognorm.rvs(s=0.5, scale=np.exp(3.6), size=n_images).clip(20, 40)  # Right-skewed, around 30dB
        elif method_rank == 5:
            vector = gamma.rvs(a=3, scale=5, size=n_images).clip(20, 35)  # More variance, centered ~25dB
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=2.3, scale=5.1, size=n_images).clip(20, 35) # More variance cent. ~25dB

        return {"psnr_enhance": vector}

    def _create_2_ssim_enhance(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_2_ssim_enhance
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=15, b=2, size=n_images)  # Peaked around 0.95-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.90, scale=0.04, size=n_images).clip(0.7, 1)  # Peaked around 0.90
        elif method_rank == 3:
            vector = beta.rvs(a=10, b=4, size=n_images)  # More variance, centered around 0.85
        elif method_rank == 4:
            vector = norm.rvs(loc=0.75, scale=0.06, size=n_images).clip(0.5, 0.9)  # Centered around 0.75
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0.5, 0.8)  # More skewed, centered around 0.65
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=3.3, scale=0.17, size=n_images).clip(0.5, 0.8) # More skewed

        return {"ssim_enhance": vector}

    def _create_3_fsim_enhance(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_3_fsim_enhance
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=16, b=2, size=n_images)  # Very high values, ~0.95-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.90, scale=0.03, size=n_images).clip(0.8, 1)  # Peaked around 0.9
        elif method_rank == 3:
            vector = beta.rvs(a=9, b=3, size=n_images)  # Skewed towards higher values, ~0.85
        elif method_rank == 4:
            vector = norm.rvs(loc=0.75, scale=0.07, size=n_images).clip(0.5, 0.9)  # More variance, centered ~0.75
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.4, scale=0.4, size=n_images).clip(0.4, 0.8)  # Wide range, centered ~0.6
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.31, scale=0.42, size=n_images).clip(0.27534, 0.8) # Wide range

        return {"fsim_enhance": vector}

    def _create_4_mse_enhance(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_4_mse_enhance
        """
        vector = []
        if method_rank == 1:
            vector = gamma.rvs(a=6, scale=0.5, size=n_images).clip(0, 10)  # Small values ~1-5
        elif method_rank == 2:
            vector = gamma.rvs(a=5, scale=1, size=n_images).clip(0, 15)  # Slightly larger values ~3-10
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.5, scale=np.exp(2), size=n_images).clip(5, 20)  # Some variation, centered around 10
        elif method_rank == 4:
            vector = gamma.rvs(a=4, scale=2, size=n_images).clip(10, 30)  # Right-skewed, centered ~15-25
        elif method_rank == 5:
            vector = expon.rvs(scale=15, size=n_images).clip(15, 50)  # Large values, long tail
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = expon.rvs(scale=20, size=n_images).clip(19, 75)  # Large values, long tail

        return {"mse_enhance": vector}

    def _create_5_nrmse_enhance(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_5_nrmse_enhance
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=15, b=3, size=n_images) * 0.2  # Peaked near 0.05-0.15
        elif method_rank == 2:
            vector = norm.rvs(loc=0.15, scale=0.03, size=n_images).clip(0.05, 0.25)  # Peaked around 0.15-0.2
        elif method_rank == 3:
            vector = beta.rvs(a=10, b=5, size=n_images) * 0.4  # More spread, centered ~0.2-0.3
        elif method_rank == 4:
            vector = lognorm.rvs(s=0.4, scale=np.exp(-1), size=n_images).clip(0.2, 0.5)  # More variation ~0.25-0.4
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.3, scale=0.5, size=n_images).clip(0.3, 0.8)  # Spread from 0.3-0.8
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.21, scale=0.44, size=n_images).clip(0.17, 0.69) # Spread 0.3-0.8

        return {"nrmse_enhance": vector}

    def _create_6_gmsd_enhance(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_6_gmsd_enhance
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=16, b=2, size=n_images) * 0.1  # Very small values ~0.02-0.08
        elif method_rank == 2:
            vector = norm.rvs(loc=0.08, scale=0.02, size=n_images).clip(0.02, 0.12)  # Peaked around 0.08
        elif method_rank == 3:
            vector = beta.rvs(a=9, b=3, size=n_images) * 0.3  # Skewed towards smaller values, ~0.1-0.2
        elif method_rank == 4:
            vector = norm.rvs(loc=0.2, scale=0.05, size=n_images).clip(0.1, 0.35)  # Centered around 0.2-0.3
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.3, scale=0.4, size=n_images).clip(0.3, 0.7)  # Wide range, centered ~0.4-0.6
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.2, scale=0.33, size=n_images).clip(0.14, 0.59) # Wide range, ~0.4-0.6

        return {"gmsd_enhance": vector}

    def _create_7_msssim_enhance(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_7_msssim_enhance
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=15, b=2, size=n_images)  # Peaked near 0.95-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.90, scale=0.04, size=n_images).clip(0.75, 1)  # Peaked around 0.9
        elif method_rank == 3:
            vector = beta.rvs(a=10, b=4, size=n_images)  # More variance, centered around 0.85
        elif method_rank == 4:
            vector = norm.rvs(loc=0.75, scale=0.06, size=n_images).clip(0.5, 0.9)  # Centered around 0.75
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0.4, 0.8)  # More skewed, centered ~0.65
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=3, scale=0.21, size=n_images).clip(0.273947, 0.66) # More skew ~0.65

        return {"msssim_enhance": vector}

    def _create_8_brisque_enhance(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_8_brisque_enhance
        """
        vector = []
        if method_rank == 1:
            vector = gamma.rvs(a=6, scale=2, size=n_images).clip(5, 20)  # Low BRISQUE values ~5-15
        elif method_rank == 2:
            vector = gamma.rvs(a=5, scale=3, size=n_images).clip(10, 30)  # Slightly larger values ~10-25
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.5, scale=np.exp(3), size=n_images).clip(15, 40)  # More variation, centered ~20-35
        elif method_rank == 4:
            vector = gamma.rvs(a=4, scale=5, size=n_images).clip(25, 50)  # Right-skewed, centered ~30-45
        elif method_rank == 5:
            vector = expon.rvs(scale=10, size=n_images).clip(40, 70)  # Large values, long tail
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = expon.rvs(scale=22, size=n_images).clip(48, 82)  # Large values, long tail

        return {"brisque_enhance": vector}

    def _create_9_nrqmniqe_enhance(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_9_nrqmniqe_enhance
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=14, b=3, size=n_images) * 2  # Peaked near 0.5-2.0
        elif method_rank == 2:
            vector = norm.rvs(loc=2.0, scale=0.5, size=n_images).clip(1, 3)  # Peaked around 2
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.3, scale=np.exp(1.2), size=n_images).clip(2, 4)  # More variation, centered around 3
        elif method_rank == 4:
            vector = beta.rvs(a=6, b=6, size=n_images) * 5  # More even spread ~2.5-5.0
        elif method_rank == 5:
            vector = uniform.rvs(loc=4, scale=5, size=n_images).clip(4, 9)  # Spread from 4 to 9
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=2.9, scale=5.4, size=n_images).clip(4, 13.74212)  # Spread

        return {"nrqmniqe_enhance": vector}

    def _create_10_wbiqi_enhance(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_10_wbiqi_enhance
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=16, b=2, size=n_images)  # Very high values, ~0.95-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.90, scale=0.03, size=n_images).clip(0.8, 1)  # Peaked around 0.9
        elif method_rank == 3:
            vector = beta.rvs(a=9, b=3, size=n_images)  # Skewed towards higher values, ~0.85
        elif method_rank == 4:
            vector = norm.rvs(loc=0.75, scale=0.07, size=n_images).clip(0.5, 0.9)  # More variance, centered ~0.75
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.4, scale=0.4, size=n_images).clip(0.4, 0.8)  # Wide range, centered ~0.6
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.22, scale=0.39, size=n_images).clip(0.26, 0.67)  # Wide range ~0.6

        return {"wbiqi_enhance": vector}

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    def _call_dabs_function(self,
                            metric : str, 
                            n_images : int, 
                            method_rank : int = 1, 
                            function : Callable = None, 
                            *args : Optional[List[Any]], 
                            **kwargs : Optional[Dict[str, Any]]):
        """
        _call_dabs_function
        """

        # New value(s)
        v = 0.0
        glitchn = np.random.randint(5, 15)

        # Function Routing
        if metric == "MOD":
            f1 = self._create_1_mod_dab
            v = list(f1(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "IOD":
            f2 = self._create_2_iod_dab
            v = list(f2(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "HEDAB":
            f3 = self._create_3_hedab_dab
            v = list(f3(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "CDE":
            f4 = self._create_4_cde_dab
            v = list(f4(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "CVAR":
            f5 = self._create_5_cvar_dab
            v = list(f5(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "SNE":
            f6 = self._create_6_sne_dab
            v = list(f6(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "ESD":
            f7 = self._create_7_esd_dab
            v = list(f7(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "BSI":
            f8 = self._create_8_bsi_dab
            v = list(f8(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "CTNR":
            f9 = self._create_9_ctnr_dab
            v = list(f9(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        else:
            raise ValueError("Invalid function option.")

        return v

    def _create_1_mod_dab(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_1_mod_dab
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=15, b=3, size=n_images)  # Strong staining, peaked near 0.85-1.0
        elif method_rank == 2:
            vector = norm.rvs(loc=0.85, scale=0.04, size=n_images).clip(0.7, 1)  # Centered ~0.85
        elif method_rank == 3:
            vector = beta.rvs(a=10, b=5, size=n_images)  # More variation, centered around 0.75
        elif method_rank == 4:
            vector = norm.rvs(loc=0.65, scale=0.08, size=n_images).clip(0.4, 0.85)  # Centered ~0.65
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0.3, 0.75)  # Skewed lower, ~0.5-0.7
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0.3, 0.75)  # Skewed lower, ~0.5-0.7

        return {"mod_dab": vector}

    def _create_2_iod_dab(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_2_iod_dab
        """
        vector = []
        if method_rank == 1:
            vector = gamma.rvs(a=8, scale=100, size=n_images).clip(500, 2000)  # High values, dense regions
        elif method_rank == 2:
            vector = gamma.rvs(a=7, scale=80, size=n_images).clip(400, 1800)  # Slightly lower but similar
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.5, scale=np.exp(6), size=n_images).clip(300, 1500)  # More variation, centered ~1000
        elif method_rank == 4:
            vector = gamma.rvs(a=5, scale=60, size=n_images).clip(200, 1200)  # Wider spread, lower values
        elif method_rank == 5:
            vector = expon.rvs(scale=250, size=n_images).clip(100, 900)  # Large variation, lower density
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = expon.rvs(scale=250, size=n_images).clip(100, 900)  # Large variation, lower dens.

        return {"iod_dab": vector}

    def _create_3_hedab_dab(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_3_hedab_dab
        """
        vector = []
        if method_rank == 1:
            vector = norm.rvs(loc=1.0, scale=0.1, size=n_images).clip(0.8, 1.2)  # Balanced ratio ~1.0
        elif method_rank == 2:
            vector = beta.rvs(a=10, b=4, size=n_images) * 1.2  # Slight skew toward 1.0-1.2
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.3, scale=np.exp(0.7), size=n_images).clip(0.5, 1.5)  # More variance, ~0.8-1.2
        elif method_rank == 4:
            vector = beta.rvs(a=6, b=6, size=n_images) * 1.5  # Wider spread, ~0.6-1.8
        elif method_rank == 5:
            vector = uniform.rvs(loc=0.3, scale=2.0, size=n_images).clip(0.3, 2.5)  # Spread
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=0.3, scale=2.0, size=n_images).clip(0.3, 2.5)  # Spread

        return {"hedab_dab": vector}

    def _create_4_cde_dab(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_4_cde_dab
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=15, b=3, size=n_images) * 0.1  # Peaked near 0.02-0.08
        elif method_rank == 2:
            vector = norm.rvs(loc=0.08, scale=0.02, size=n_images).clip(0.03, 0.12)  # Centered ~0.08
        elif method_rank == 3:
            vector = beta.rvs(a=10, b=5, size=n_images) * 0.2  # More variation, centered around 0.1-0.15
        elif method_rank == 4:
            vector = norm.rvs(loc=0.15, scale=0.05, size=n_images).clip(0.08, 0.25)  # Centered ~0.15-0.2
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.05, size=n_images).clip(0.1, 0.3)  # Right-skewed, ~0.15-0.3
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=4, scale=0.05, size=n_images).clip(0.1, 0.3)  # Right-skewed, ~0.15-0.3

        return {"cde_dab": vector}

    def _create_5_cvar_dab(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_5_cvar_dab
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=15, b=3, size=n_images) * 0.3  # Highly uniform, mostly 0.1-0.2
        elif method_rank == 2:
            vector = norm.rvs(loc=0.2, scale=0.05, size=n_images).clip(0.1, 0.35)  # Slightly more variation
        elif method_rank == 3:
            vector = beta.rvs(a=8, b=4, size=n_images) * 0.5  # More variation, centered around 0.3
        elif method_rank == 4:
            vector = norm.rvs(loc=0.5, scale=0.1, size=n_images).clip(0.3, 0.8)  # Centered around 0.5
        elif method_rank == 5:
            vector = gamma.rvs(a=3, scale=0.3, size=n_images).clip(0.4, 1.5)  # Wider range, poor uniformity
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=3, scale=0.3, size=n_images).clip(0.4, 1.5)  # Wider range

        return {"cvar_dab": vector}

    def _create_6_sne_dab(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_6_sne_dab
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=16, b=2, size=n_images) * 0.05  # Very low error, ~0.01-0.04
        elif method_rank == 2:
            vector = norm.rvs(loc=0.05, scale=0.02, size=n_images).clip(0.02, 0.08)  # Centered ~0.05
        elif method_rank == 3:
            vector = beta.rvs(a=10, b=4, size=n_images) * 0.1  # More variation, centered around 0.05-0.1
        elif method_rank == 4:
            vector = norm.rvs(loc=0.15, scale=0.05, size=n_images).clip(0.08, 0.2)  # Centered ~0.15
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.05, size=n_images).clip(0.1, 0.3)  # Right-skewed, ~0.15-0.3
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=4, scale=0.05, size=n_images).clip(0.1, 0.3)  # ~0.15-0.3

        return {"sne_dab": vector}

    def _create_7_esd_dab(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_7_esd_dab
        """
        vector = []
        if method_rank == 1:
            vector = beta.rvs(a=15, b=3, size=n_images) * 0.2  # Very low entropy, ~0.02-0.15
        elif method_rank == 2:
            vector = norm.rvs(loc=0.15, scale=0.03, size=n_images).clip(0.05, 0.25)  # Centered ~0.15
        elif method_rank == 3:
            vector = beta.rvs(a=10, b=5, size=n_images) * 0.3  # More variation, centered around 0.2
        elif method_rank == 4:
            vector = norm.rvs(loc=0.25, scale=0.07, size=n_images).clip(0.1, 0.4)  # Centered ~0.25
        elif method_rank == 5:
            vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0.2, 0.5)  # Right-skewed, ~0.3-0.5
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = gamma.rvs(a=4, scale=0.1, size=n_images).clip(0.2, 0.5)  # Right-skewed, ~0.3-0.5

        return {"esd_dab": vector}

    def _create_8_bsi_dab(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_8_bsi_dab
        """
        vector = []
        if method_rank == 1:
            vector = gamma.rvs(a=6, scale=3, size=n_images).clip(0, 25)  # Low background, mostly ~5-20
        elif method_rank == 2:
            vector = gamma.rvs(a=5, scale=4, size=n_images).clip(10, 40)  # Slightly higher but reasonable
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.5, scale=np.exp(3.5), size=n_images).clip(20, 60) # More variation, around 35-50
        elif method_rank == 4:
            vector = gamma.rvs(a=4, scale=6, size=n_images).clip(30, 80)  # Wider spread, higher background intensity
        elif method_rank == 5:
            vector = expon.rvs(scale=20, size=n_images).clip(50, 120)  # High values, long tail
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = expon.rvs(scale=20, size=n_images).clip(50, 120)  # High values, long tail

        return {"bsi_dab": vector}

    def _create_9_ctnr_dab(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_9_ctnr_dab
        """
        vector = []
        if method_rank == 1:
            vector = norm.rvs(loc=90, scale=5, size=n_images).clip(70, 100)  # Very high contrast, little noise
        elif method_rank == 2:
            vector = beta.rvs(a=14, b=3, size=n_images) * 20 + 70  # Peaked around 75-85
        elif method_rank == 3:
            vector = lognorm.rvs(s=0.3, scale=np.exp(3.5), size=n_images).clip(50, 80)  # More variation, centered ~65-75
        elif method_rank == 4:
            vector = beta.rvs(a=6, b=6, size=n_images) * 50 + 30  # More even spread ~40-65
        elif method_rank == 5:
            vector = uniform.rvs(loc=20, scale=40, size=n_images).clip(20, 60)  # Spread from 20 to 60
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = uniform.rvs(loc=20, scale=40, size=n_images).clip(20, 60)  # Spread from 20 to 60

        return {"ctnr_dab": vector}


    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    def _call_hard_function(self, 
                            metric : str, 
                            n_images : int, 
                            method_rank : int = 1, 
                            function : Callable = None, 
                            *args : Optional[List[Any]], 
                            **kwargs : Optional[Dict[str, Any]]):
        """
        _call_hard_function
        """

        # New value(s)
        v = 0.0
        glitchn = np.random.randint(5, 15)

        # Function Routing
        if metric == "IT": # TODO
            f1 = self._create_1_it_hardware
            v = list(f1(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "TPT":
            f2 = self._create_2_tpt_hardware
            v = list(f2(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "VRAM":
            f3 = self._create_3_vram_hardware
            v = list(f3(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "RAM":
            f4 = self._create_4_ram_hardware
            v = list(f4(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "MACs":
            f5 = self._create_5_macs_hardware
            v = list(f5(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "BW":
            f6 = self._create_6_bw_hardware
            v = list(f6(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "EPI":
            f7 = self._create_7_epi_hardware
            v = list(f7(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "TTA":
            f8 = self._create_8_tta_hardware
            v = list(f8(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "LAT":
            f9 = self._create_9_lat_hardware
            v = list(f9(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "MTTF":
            f10 = self._create_10_mttf_hardware
            v = list(f10(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        else:
            raise ValueError("Invalid function option.")

        return v

    def _create_1_it_hardware(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_1_it_hardware
        """

        vector = []
        if method_rank == 1:
            vector = np.random.gamma(shape=2, scale=0.02, size=n_images)
        elif method_rank == 2:
            vector = np.random.gamma(shape=2, scale=0.03, size=n_images)
        elif method_rank == 3:
            vector = np.random.gamma(shape=2, scale=0.04, size=n_images)
        elif method_rank == 4:
            vector = np.random.lognormal(mean=0.02, sigma=0.4, size=n_images)
        elif method_rank == 5:
            vector = np.random.lognormal(mean=0.03, sigma=0.5, size=n_images)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = np.random.lognormal(mean=0.03, sigma=0.5, size=n_images)

        return {"it_hardware": vector}

    def _create_2_tpt_hardware(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_2_tpt_hardware
        """

        vector = []
        if method_rank == 1:
            vector = np.random.normal(loc=500, scale=50, size=n_images)  # High mean, small variance
        elif method_rank == 2:
            vector = np.random.normal(loc=400, scale=50, size=n_images)
        elif method_rank == 3:
            vector = np.random.normal(loc=300, scale=60, size=n_images)
        elif method_rank == 4:
            vector = np.random.poisson(lam=200, size=n_images)  # Poisson for sporadic batch sizes
        elif method_rank == 5:
            vector = np.random.poisson(lam=100, size=n_images)  # More jitter and lower mean
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = np.random.poisson(lam=100, size=n_images)  # More jitter and lower mean

        return {"tpt_hardware": vector}

    def _create_3_vram_hardware(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_3_vram_hardware
        """

        vector = []
        if method_rank == 1:
            vector = np.clip(np.random.normal(loc=8000, scale=500, size=n_images), 7000, 9000)
        elif method_rank == 2:
            vector = np.clip(np.random.normal(loc=10000, scale=600, size=n_images), 8500, 11500)
        elif method_rank == 3:
            vector = np.clip(np.random.normal(loc=12000, scale=800, size=n_images), 10000, 14000)
        elif method_rank == 4:
            vector = np.clip(np.random.normal(loc=16000, scale=1200, size=n_images), 13000, 20000)
        elif method_rank == 5:
            vector = np.clip(np.random.exponential(scale=3000, size=n_images) + 16000, 16000, 25000)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = np.clip(np.random.exponential(scale=3000, size=n_images) + 16000, 16000, 25000)

        return {"vram_hardware": vector}

    def _create_4_ram_hardware(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_4_ram_hardware
        """

        vector = []
        if method_rank == 1:
            vector = np.clip(np.random.normal(loc=8000, scale=500, size=n_images), 7000, 9000)
        elif method_rank == 2:
            vector = np.clip(np.random.normal(loc=10000, scale=600, size=n_images), 8500, 11500)
        elif method_rank == 3:
            vector = np.clip(np.random.normal(loc=12000, scale=800, size=n_images), 10000, 14000)
        elif method_rank == 4:
            vector = np.clip(np.random.lognormal(mean=9, sigma=0.4, size=n_images), 13000, 18000)
        elif method_rank == 5:
            vector = np.clip(np.random.lognormal(mean=9.5, sigma=0.5, size=n_images), 16000, 25000)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = np.clip(np.random.lognormal(mean=9.5, sigma=0.5, size=n_images), 16000, 25000)

        return {"ram_hardware": vector}

    def _create_5_macs_hardware(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_5_macs_hardware
        """

        vector = []
        if method_rank == 1:
            vector = np.random.gamma(shape=2, scale=5e11, size=n_images)
        elif method_rank == 2:
            vector = np.random.gamma(shape=2, scale=7e11, size=n_images)
        elif method_rank == 3:
            vector = np.random.gamma(shape=2, scale=1e12, size=n_images)
        elif method_rank == 4:
            vector = np.random.exponential(scale=2e12, size=n_images)
        elif method_rank == 5:
            vector = np.random.exponential(scale=3e12, size=n_images)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = np.random.exponential(scale=3e12, size=n_images)

        return {"macs_hardware": vector}

    def _create_6_bw_hardware(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_6_bw_hardware
        """

        vector = []
        if method_rank == 1:
            vector = np.clip(np.random.normal(loc=500, scale=50, size=n_images), 400, 600)
        elif method_rank == 2:
            vector = np.clip(np.random.normal(loc=400, scale=50, size=n_images), 320, 500)
        elif method_rank == 3:
            vector = np.clip(np.random.normal(loc=300, scale=60, size=n_images), 200, 400)
        elif method_rank == 4:
            vector = np.clip(500 * np.random.beta(a=2, b=5, size=n_images), 100, 350)
        elif method_rank == 5:
            vector = np.clip(600 * np.random.beta(a=1.5, b=4, size=n_images), 50, 300)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = np.clip(600 * np.random.beta(a=1.5, b=4, size=n_images), 50, 300)

        return {"bw_hardware": vector}

    def _create_7_epi_hardware(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_7_epi_hardware
        """

        vector = []
        if method_rank == 1:
            vector = np.clip(np.random.exponential(scale=2, size=n_images), 1, 5)
        elif method_rank == 2:
            vector = np.clip(np.random.exponential(scale=3, size=n_images), 2, 7)
        elif method_rank == 3:
            vector = np.clip(np.random.normal(loc=5, scale=2, size=n_images), 3, 10)
        elif method_rank == 4:
            vector = np.clip(np.random.normal(loc=10, scale=3, size=n_images), 5, 15)
        elif method_rank == 5:
            vector = np.clip(np.random.exponential(scale=8, size=n_images), 10, 30)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = np.clip(np.random.exponential(scale=8, size=n_images), 10, 30)

        return {"epi_hardware": vector}

    def _create_8_tta_hardware(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_8_tta_hardware
        """

        vector = []
        if method_rank == 1:
            vector = np.random.gamma(shape=2, scale=1, size=n_images)
        elif method_rank == 2:
            vector = np.random.gamma(shape=2, scale=2, size=n_images)
        elif method_rank == 3:
            vector = np.random.gamma(shape=2, scale=5, size=n_images)
        elif method_rank == 4:
            vector = np.random.lognormal(mean=1.5, sigma=0.6, size=n_images)
        elif method_rank == 5:
            vector = np.random.lognormal(mean=2, sigma=0.8, size=n_images)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = np.random.lognormal(mean=2, sigma=0.8, size=n_images)

        return {"tta_hardware": vector}

    def _create_9_lat_hardware(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_9_lat_hardware
        """

        vector = []
        if method_rank == 1:
            vector = np.random.gamma(shape=2, scale=5, size=n_images)
        elif method_rank == 2:
            vector = np.random.gamma(shape=2, scale=8, size=n_images)
        elif method_rank == 3:
            vector = np.random.gamma(shape=2, scale=12, size=n_images)
        elif method_rank == 4:
            vector = np.random.weibull(a=1.5, size=n_images) * 20
        elif method_rank == 5:
            vector = np.random.weibull(a=1.2, size=n_images) * 40
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = np.random.weibull(a=1.2, size=n_images) * 40

        return {"lat_hardware": vector}

    def _create_10_mttf_hardware(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_10_mttf_hardware
        """

        vector = []
        if method_rank == 1:
            vector = np.random.normal(loc=5000, scale=500, size=n_images)
        elif method_rank == 2:
            vector = np.random.normal(loc=4000, scale=600, size=n_images)
        elif method_rank == 3:
            vector = np.random.normal(loc=3000, scale=800, size=n_images)
        elif method_rank == 4:
            vector = np.random.weibull(a=1.5, size=n_images) * 2000
        elif method_rank == 5:
            vector = np.random.weibull(a=1.2, size=n_images) * 1500
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = np.random.weibull(a=1.2, size=n_images) * 1500

        return {"mttf_hardware": vector}

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    def number_of_digits(self, n):
        """
        Return the number of digits of n
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        return int(math.log10(n)) + 1

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

    def _add_glitch(self, value, n, maxv=1.0, minv=0.0, mult=1):
        """
        add_glitch
        """

        possibility_vector = np.array(["mili", "micro", "nano", "pico", "femto",
                                       "atto", "zepto", "yocto", "ronto", "quecto"])
        sign_vector = np.array([-1, 1])

        for i in range(0, n):

            item = np.random.choice(possibility_vector)
            sign = np.random.choice(sign_vector)
            value += (self._glitch(0.0, type=item) * sign * mult)

        return max(min(value, maxv), minv)

    def _glitch(self, value, type="pico", function=None, *args, **kwargs):
        """
        # Uniform
        from scipy.stats import uniform
        num = uniform.rvs(loc=0, scale=10)  # loc = start, scale = range width

        # Normal Distribution (Gaussian)
        import numpy as np
        num = np.random.normal(loc=5, scale=2)  # Mean 5, Std 2
        num = np.clip(num, 0, 10)  # Ensure values are inside [0,10]

        # Custom probability function
        choices = np.linspace(0, 10, 1000)  # 1000 points from 0 to 10
        probabilities = np.exp(-choices)  # Example: exponential decay
        probabilities /= probabilities.sum()  # Normalize to sum to 1
        num = np.random.choice(choices, p=probabilities)

        ---------------------------------------------------------------

        1. Understanding *args:
        *args collects all extra positional arguments passed to a function.
        It stores them as a tuple.
        Example:
        def example_function(*args):
            print(args)  # args is a tuple
        example_function(1, 2, 3, "hello")  
        # Output: (1, 2, 3, 'hello')

        2. Understanding **kwargs:
        **kwargs collects all extra named arguments.
        It stores them as a dictionary (dict).
        Example:
        def example_function(**kwargs):
            print(kwargs)  # kwargs is a dictionary
        example_function(a=1, b=2, c="hello")  
        # Output: {'a': 1, 'b': 2, 'c': 'hello'}

        3. Mixing Both
        def my_function(x, *args, **kwargs):
            print(f"x: {x}")
            print(f"args: {args}")
            print(f"kwargs: {kwargs}")
        my_function(10, "extra1", "extra2", key1="value1", key2="value2")
        """

        random_digit = None
        if function:
            random_digit = function(*args, **kwargs)
            random_digit = np.clip(random_digit, 0, 10)
        else:
            random_digit = np.random.randint(0, 10)  # Random number from 0 to 9

        if type=="mili":
            new_n = int(value * 10) / 10 + random_digit / 1e1
        elif type=="micro":
            new_n = int(value * 10) / 10 + random_digit / 1e2
        elif type=="nano":
            new_n = int(value * 10) / 10 + random_digit / 1e3
        elif type=="pico":
            new_n = int(value * 10) / 10 + random_digit / 1e4
        elif type=="femto":
            new_n = int(value * 10) / 10 + random_digit / 1e5
        elif type=="atto":
            new_n = int(value * 10) / 10 + random_digit / 1e6
        elif type=="zepto":
            new_n = int(value * 10) / 10 + random_digit / 1e7
        elif type=="yocto":
            new_n = int(value * 10) / 10 + random_digit / 1e8
        elif type=="ronto":
            new_n = int(value * 10) / 10 + random_digit / 1e9
        elif type=="quecto":
            new_n = int(value * 10) / 10 + random_digit / 1e10
        else:
            new_n = value

        return new_n

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

    def _fix_vector_length(self, final_vector : List[int], total : int) -> List[int]:
        """
        fix_vector_length post-hoc size fix
        """

        current_length = len(final_vector)
        diff = total - current_length

        # Nothing to fix
        if diff == 0:
            return final_vector

        counts = Counter(final_vector)
        values, freqs = zip(*counts.items())

        # Convert counts to proportions
        total_current = sum(freqs)
        proportions = [f / total_current for f in freqs]

        if diff > 0:

            # Need to add 'diff' elements, preserving proportions
            additions = random.choices(values, weights=proportions, k=diff)
            final_vector.extend(additions)

        elif diff < 0:
            # Need to remove '-diff' elements, preserving proportions

            # Strategy: delete values proportional to their frequency
            to_remove = Counter(random.choices(final_vector, k=-diff))

            new_vector = []
            for val in final_vector:
                if to_remove[val] > 0:
                    to_remove[val] -= 1
                else:
                    new_vector.append(val)
            final_vector = new_vector

        return final_vector

    def _appendage(self, n : int, proportions : List[int]) -> List[int]:
        """
        Append values to dictionary given proportions
        """
        flist = [v for i, p in enumerate(proportions) for v in [i + 1] * int((p / 100) * n)]
        return self._fix_vector_length(flist, n)

    def _calculate_real_rank_list(self, methods_dict : Dict[str, int], n_images : int) -> Dict[str, List]:
        """
        fix_vector_length post-hoc size fix
        """

        # Creating real rank dictionary
        final_dict = OrderedDict()

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

        # Iterating on proportions and methods dictionaries
        for method_name, method_rank in methods_dict.items():
            prop_vec = proportions_dict[method_rank]
            final_dict[method_name] = self._appendage(n_images, prop_vec)

        # Return real rank dictionary
        return final_dict
