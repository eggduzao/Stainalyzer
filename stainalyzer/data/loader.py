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
from typing import Any, List, Dict, Optional, Callable, Union, Tuple
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

from stainalyzer.data.proportions import Proportions


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
    1. [F1SEG]   Dice Coefficient (F1-Score for Segmentation)
    2. [IoU]     Intersection over Union (IoU, Jaccard Index)
    3. [PxAcc]   Pixel-wise Accuracy
    4. [AJI]     Aggregated Jaccard Index
    5. [HD95]    Hausdorff Distance (HD95, 95th percentile)
    6. [F1BDR]   Boundary F1-Score (BF1)
    7. [F1DET]   Object Detection F1-Score (F1-Detect)
    8. [PQ]      Panoptic Quality (PQ)
    9. [ARI]     Adjusted Rand Index (ARI)
    10. [RECSEG] Segmentation Recall
    11. [PRESEG] Segmentation Precision

    B: Staining Quantification
    1. [MOD]   Mean Optical Density (MOD)
    2. [IOD]   Integrated Optical Density (IOD)
    3. [HEDAB] Hematoxylin-DAB Ratio
    4. [CDE]   Color Deconvolution Error
    5. [CVAR]  Coefficient of Variation (CV)
    6. [SNE]   Stain Normalization Error
    7. [ESD]   Entropy of Staining Distribution
    8. [BSI]   Background Stain Intensity
    9. [CTNR]  Contrast-to-Noise Ratio (CNR)

    C: Hardware                               
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

    D: Classification
    1. [CMA]   Confusion Matrix
    2. [F1S]   F1-Score (Classification)
    3. [FBET]  F-beta Score (Classification)
    4. [BSCO]  Brier Score (Calibration Error)
    5. [CROE]  Normalized Cross Entropy
    6. [COHK]  Cohen’s Kappa (Inter-rater Agreement)
    7. [SPE]   Specificity (True Negative Rate)
    8. [AUROC] Area Under the ROC Curve (AUC-ROC)
    9. [AUPR]  Area Under the Precision-Recall Curve (AUC-PR)
    10. [MCC]  Matthews Correlation Coefficient (MCC)

    E: All
    1. [CWAV] Weighted average of all A scores.
    2. [SWAV] Weighted average of all B scores.
    3. [HWAV] Weighted average of all C scores.
    4. [LWAV] Weighted average of all D scores.
    5. [CSWA] Weighted average between (A + B).
    6. [CSHWA] Weighted average between (A + B + C).
    7. [CSHLWA] Weighted average between (A + B + C + D).
    """

    def __init__(self):

        # All Segmentation (Cell and nucleus) Methods
        self.segmentation_methods = OrderedDict()
        self.segmentation_methods["nnU_Net"] = 1
        self.segmentation_methods["Cellpose2"] = 2
        self.segmentation_methods["Mesmer"] = 3
        self.segmentation_methods["Omnipose"] = 4
        self.segmentation_methods["LumenNet"] = 5
        self.segmentation_methods["StarDist"] = 6
        self.segmentation_methods["SAM"] = 7 # Segment Anything
        self.segmentation_methods["Cellpose"] = 8
        self.segmentation_methods["Ilastik"] = 9
        self.segmentation_methods["HoVer_Net"] = 10
        self.segmentation_methods["DeepLabV3P"] = 11
        self.segmentation_methods["CellProfiler4"] = 12
        self.segmentation_methods["UNet"] = 13
        self.segmentation_methods["Att_UNet"] = 14
        self.segmentation_methods["SAP_UNet"] = 15
        self.segmentation_methods["LUNet"] = 16
        self.segmentation_methods["UNet_SSTBM"] = 17
        self.segmentation_methods["Swin_Transformer"] = 18
        self.segmentation_methods["Swin_V2"] = 19
        self.segmentation_methods["ViT_MoE"] = 20
        self.segmentation_methods["SNOW"] = 21
        self.segmentation_methods["DINOv2_ViT_L14"] = 22
        self.segmentation_methods["DINO_ViT_B16"] = 23
        self.segmentation_methods["LeWin"] = 24 # LeWinSpectralTransformer
        self.segmentation_methods["DINO_ViT_B8"] = 25
        self.segmentation_methods["DINO_ViT_S16"] = 26
        self.segmentation_methods["DINO_ViT_S8"] = 27
        self.segmentation_methods["SimCLR"] = 28
        self.segmentation_methods["MoCo"] = 29
        self.segmentation_methods["uniDINO"] = 30
        self.segmentation_methods["VAE"] = 31
        self.segmentation_methods["DeepCell"] = 32
        self.segmentation_methods["MitoSSL"] = 33
        self.segmentation_methods["MyoFuse"] = 34
        self.segmentation_methods["Watershed"] = 35
        self.segmentation_methods["ISC_GAN"] = 36
        self.segmentation_methods["Fiji_DeepImageJ"] = 37
        self.segmentation_methods["Piximi"] = 38
        self.segmentation_methods["QuPath"] = 39
        self.segmentation_methods["DINO"] = 40
        self.segmentation_methods["CellSeg"] = 41
        self.segmentation_methods["ResNet"] = 42
        self.segmentation_methods["EpidermaQuant"] = 43
        self.segmentation_methods["HistomicsTK"] = 44
        self.segmentation_methods["IHC_Profiler"] = 45
        self.segmentation_methods["DABQuant"] = 46
        self.segmentation_methods["scikit_image"] = 47
        self.segmentation_methods["OsiLab"] = 48
        self.segmentation_methods["Patech"] = 49
        self.segmentation_methods["CA_SSL"] = 50
        self.segmentation_methods["CelloType"] = 51
        self.segmentation_methods["Sribdmed"] = 52
        self.segmentation_methods["Otsu"] = 53
        self.segmentation_methods["Sanmed_AI"] = 54
        self.segmentation_methods["CEMotate_V3"] = 55
        self.segmentation_methods["GuanLab"] = 56
        self.segmentation_methods["ArontierHYY"] = 57
        self.segmentation_methods["HDPMM"] = 58 # Hierarchical Dirichlet Processes Mixture Model
        self.segmentation_methods["NucCell_GAN"] = 59
        self.segmentation_methods["VGG_19"] = 60
        self.segmentation_methods["ImageJ"] = 61
        self.segmentation_methods["HiLab"] = 62
        self.segmentation_methods["RGrowing"] = 63
        self.segmentation_methods["Vahid"] = 64
        self.segmentation_methods["GAN"] = 65
        self.segmentation_methods["Jorey"] = 66
        self.segmentation_methods["Random_Forest"] = 67
        self.segmentation_methods["SVM"] = 68
        self.segmentation_methods["KIT_GE3"] = 69
        self.segmentation_methods["RealEsrgan"] = 70 # RealESRGAN_x4plus
        self.segmentation_methods["StableDiffusion"] = 71
        self.segmentation_methods["ZoeDepth"] = 72
        self.segmentation_methods["AIExplore"] = 73
        self.segmentation_methods["daschlab"] = 74
        self.segmentation_methods["Turbolag"] = 75
        self.segmentation_methods["CMIAS"] = 76

        # Staining Methods
        self.stain_methods = OrderedDict()
        self.stain_methods["HistomicsTK"] = 1
        self.stain_methods["LumenNet"] = 2
        self.stain_methods["QuPath"] = 3
        self.stain_methods["CellProfiler4"] = 4
        self.stain_methods["IHC_Profiler"] = 5
        self.stain_methods["EpidermaQuant"] = 6
        self.stain_methods["DABQuant"] = 7
        self.stain_methods["Fiji_DeepImageJ"] = 8
        self.stain_methods["Cellpose2"] = 9
        self.stain_methods["DeepCell"] = 10
        self.stain_methods["ImageJ"] = 11
        self.stain_methods["Mesmer"] = 12
        self.stain_methods["Cellpose"] = 13
        self.stain_methods["UNet"] = 14
        self.stain_methods["nnU_Net"] = 15
        self.stain_methods["HoVer_Net"] = 16
        self.stain_methods["UNet_SSTBM"] = 17
        self.stain_methods["SAP_UNet"] = 18
        self.stain_methods["Att_UNet"] = 19
        self.stain_methods["SAM"] = 20
        self.stain_methods["StarDist"] = 21
        self.stain_methods["Omnipose"] = 22
        self.stain_methods["DeepLabV3P"] = 23
        self.stain_methods["MyoFuse"] = 24
        self.stain_methods["MitoSSL"] = 25
        self.stain_methods["DINOv2_ViT_L14"] = 26
        self.stain_methods["Swin_Transformer"] = 27
        self.stain_methods["DINO_ViT_B16"] = 28
        self.stain_methods["Swin_V2"] = 29
        self.stain_methods["DINO_ViT_B8"] = 30
        self.stain_methods["DINO_ViT_S16"] = 31
        self.stain_methods["DINO_ViT_S8"] = 32
        self.stain_methods["SimCLR"] = 33
        self.stain_methods["MoCo"] = 34
        self.stain_methods["uniDINO"] = 35
        self.stain_methods["LeWin"] = 36
        self.stain_methods["VAE"] = 37
        self.stain_methods["LUNet"] = 38
        self.stain_methods["ViT_MoE"] = 39
        self.stain_methods["ResNet"] = 40
        self.stain_methods["VGG_19"] = 41
        self.stain_methods["Random_Forest"] = 42
        self.stain_methods["SVM"] = 43
        self.stain_methods["SNOW"] = 44
        self.stain_methods["RGrowing"] = 45
        self.stain_methods["scikit_image"] = 46
        self.stain_methods["HDPMM"] = 47
        self.stain_methods["Otsu"] = 48
        self.stain_methods["Ilastik"] = 49
        self.stain_methods["StableDiffusion"] = 50
        self.stain_methods["RealEsrgan"] = 51
        self.stain_methods["ZoeDepth"] = 52
        self.stain_methods["GAN"] = 53
        self.stain_methods["ISC_GAN"] = 54
        self.stain_methods["DINO"] = 55
        self.stain_methods["NucCell_GAN"] = 56
        self.stain_methods["AIExplore"] = 57
        self.stain_methods["CA_SSL"] = 58
        self.stain_methods["Sribdmed"] = 59
        self.stain_methods["Sanmed_AI"] = 60
        self.stain_methods["GuanLab"] = 61
        self.stain_methods["daschlab"] = 62
        self.stain_methods["CMIAS"] = 63
        self.stain_methods["HiLab"] = 64
        self.stain_methods["Piximi"] = 65
        self.stain_methods["Patech"] = 66
        self.stain_methods["Jorey"] = 67
        self.stain_methods["Vahid"] = 68
        self.stain_methods["KIT_GE3"] = 69
        self.stain_methods["OsiLab"] = 70
        self.stain_methods["Turbolag"] = 71
        self.stain_methods["CelloType"] = 72
        self.stain_methods["CellSeg"] = 73
        self.stain_methods["Watershed"] = 74
        self.stain_methods["ArontierHYY"] = 75
        self.stain_methods["CEMotate_V3"] = 76

        # Hardware Methods
        self.hardware_methods = OrderedDict()
        self.hardware_methods["GhostNet"] = 1
        self.hardware_methods["Fiji_DeepImageJ"] = 2
        self.hardware_methods["EpidermaQuant"] = 3
        self.hardware_methods["HistomicsTK"] = 4
        self.hardware_methods["DAB_quant"] = 5
        self.hardware_methods["Watershed"] = 6
        self.hardware_methods["Otsu"] = 7
        self.hardware_methods["DeepCell"] = 8
        self.hardware_methods["UNet_SSTBM"] = 9
        self.hardware_methods["QuPath"] = 10
        self.hardware_methods["Att_UNet"] = 11
        self.hardware_methods["UNet"] = 12
        self.hardware_methods["DeepLabV3P"] = 13
        self.hardware_methods["Cellpose"] = 14
        self.hardware_methods["ilastik"] = 15
        self.hardware_methods["Fiji_DeepImageJ"] = 16
        self.hardware_methods["HistomicsTK"] = 17
        self.hardware_methods["CellProfler4"] = 18
        self.hardware_methods["KIT_GE3"] = 19
        self.hardware_methods["Piximi"] = 20
        self.hardware_methods["CelloType"] = 21
        self.hardware_methods["SAM"] = 22
        self.hardware_methods["SAP_UNet"] = 23
        self.hardware_methods["LUNet"] = 24
        self.hardware_methods["StableDiffusion"] = 25
        self.hardware_methods["RealEsrgan"] = 26
        self.hardware_methods["ZoeDepth"] = 27
        self.hardware_methods["ViT_MoE"] = 28
        self.hardware_methods["NucCell_GAN"] = 29
        self.hardware_methods["Swin_V2"] = 30
        self.hardware_methods["SNOW"] = 31
        self.hardware_methods["osilab"] = 32
        self.hardware_methods["EpidermaQuant"] = 33
        self.hardware_methods["PATECH"] = 34
        self.hardware_methods["scikit_image"] = 35
        self.hardware_methods["Byungjae Lee"] = 36
        self.hardware_methods["DAB_quant"] = 37
        self.hardware_methods["Watershed"] = 38
        self.hardware_methods["sribdmed"] = 39
        self.hardware_methods["cells"] = 40
        self.hardware_methods["saltfish"] = 41
        self.hardware_methods["Turbolag"] = 42
        self.hardware_methods["Otsu"] = 43
        self.hardware_methods["Random_Forest"] = 44
        self.hardware_methods["ArontierHYY"] = 45
        self.hardware_methods["Newhyun00"] = 46
        self.hardware_methods["CMIAS"] = 47
        self.hardware_methods["SVM"] = 48
        self.hardware_methods["redcat_autox"] = 49
        self.hardware_methods["Jorey"] = 50
        self.hardware_methods["AIExplore"] = 51
        self.hardware_methods["Skyuser"] = 52
        self.hardware_methods["train4ever"] = 53
        self.hardware_methods["Vahid"] = 54
        self.hardware_methods["overoverfitting"] = 55
        self.hardware_methods["vipa"] = 56
        self.hardware_methods["naf"] = 57
        self.hardware_methods["bupt_mcprl"] = 58
        self.hardware_methods["cphitsz"] = 59
        self.hardware_methods["wonderworker"] = 60
        self.hardware_methods["cvmli"] = 61
        self.hardware_methods["m1n9x"] = 62
        self.hardware_methods["fzu312"] = 63
        self.hardware_methods["sgroup"] = 64
        self.hardware_methods["smf"] = 65
        self.hardware_methods["sanmed_ai"] = 66
        self.hardware_methods["hilab"] = 67
        self.hardware_methods["guanlab"] = 68
        self.hardware_methods["daschlab"] = 69
        self.hardware_methods["mbzuai_cellseg"] = 70
        self.hardware_methods["quiil"] = 71
        self.hardware_methods["plf"] = 72
        self.hardware_methods["siatcct"] = 73
        self.hardware_methods["nonozz"] = 74
        self.hardware_methods["littlefatfish"] = 75
        self.hardware_methods["littlefatfish"] = 76

        # All Methods
        self.all_methods = OrderedDict()
        self.all_methods["LumenNet"] = 1
        self.all_methods["QuPath"] = 2
        self.all_methods["CellProfiler4"] = 3
        self.all_methods["HistomicsTK"] = 4
        self.all_methods["Fiji_DeepImageJ"] = 5
        self.all_methods["ImageJ"] = 6
        self.all_methods["Cellpose2"] = 7
        self.all_methods["Cellpose"] = 8
        self.all_methods["HoVer_Net"] = 9
        self.all_methods["StarDist"] = 10
        self.all_methods["UNet_SSTBM"] = 11
        self.all_methods["UNet"] = 12
        self.all_methods["Mesmer"] = 13
        self.all_methods["IHC_Profiler"] = 14
        self.all_methods["nnU_Net"] = 15
        self.all_methods["LUNet"] = 16
        self.all_methods["Otsu"] = 17
        self.all_methods["DeepLabV3P"] = 18
        self.all_methods["DeepCell"] = 19
        self.all_methods["MyoFuse"] = 20
        self.all_methods["SAP_UNet"] = 21
        self.all_methods["SAM"] = 22
        self.all_methods["MoCo"] = 23
        self.all_methods["SimCLR"] = 24
        self.all_methods["Omnipose"] = 25
        self.all_methods["MitoSSL"] = 26
        self.all_methods["uniDINO"] = 27
        self.all_methods["DINOv2_ViT_L14"] = 28
        self.all_methods["DINO_ViT_B16"] = 29
        self.all_methods["Att_UNet"] = 30
        self.all_methods["DINO_ViT_S16"] = 31
        self.all_methods["Swin_Transformer"] = 32
        self.all_methods["Swin_V2"] = 33
        self.all_methods["DINO_ViT_B8"] = 34
        self.all_methods["ResNet"] = 35
        self.all_methods["DINO_ViT_S8"] = 36
        self.all_methods["VGG_19"] = 37
        self.all_methods["ViT_MoE"] = 38
        self.all_methods["LeWin"] = 39
        self.all_methods["DABQuant"] = 40
        self.all_methods["RealEsrgan"] = 41
        self.all_methods["StableDiffusion"] = 42
        self.all_methods["GAN"] = 43
        self.all_methods["VAE"] = 44
        self.all_methods["RGrowing"] = 45
        self.all_methods["Random_Forest"] = 46
        self.all_methods["SVM"] = 47
        self.all_methods["SNOW"] = 48
        self.all_methods["HDPMM"] = 49
        self.all_methods["scikit_image"] = 50
        self.all_methods["ISC_GAN"] = 51
        self.all_methods["DINO"] = 52
        self.all_methods["EpidermaQuant"] = 53
        self.all_methods["NucCell_GAN"] = 54
        self.all_methods["Ilastik"] = 55
        self.all_methods["Piximi"] = 56
        self.all_methods["Sribdmed"] = 57
        self.all_methods["daschlab"] = 58
        self.all_methods["Sanmed_AI"] = 59
        self.all_methods["Turbolag"] = 60
        self.all_methods["Watershed"] = 61
        self.all_methods["OsiLab"] = 62
        self.all_methods["KIT_GE3"] = 63
        self.all_methods["Patech"] = 64
        self.all_methods["Jorey"] = 65
        self.all_methods["GuanLab"] = 66
        self.all_methods["ZoeDepth"] = 67
        self.all_methods["AIExplore"] = 68
        self.all_methods["CA_SSL"] = 69
        self.all_methods["CelloType"] = 70
        self.all_methods["CellSeg"] = 71
        self.all_methods["ArontierHYY"] = 72
        self.all_methods["CEMotate_V3"] = 73
        self.all_methods["CMIAS"] = 74
        self.all_methods["Vahid"] = 75
        self.all_methods["HiLab"] = 76

        # Selected Methods
        self.selected_methods = OrderedDict()
        self.selected_methods["LumenNet"] = 1
        self.selected_methods["2"] = 2
        self.selected_methods["3"] = 3
        self.selected_methods["4"] = 4
        self.selected_methods["5"] = 5
        self.selected_methods["6"] = 6
        self.selected_methods["7"] = 7
        self.selected_methods["8"] = 8
        self.selected_methods["9"] = 9
        self.selected_methods["0"] = 10

        # Datasets
        self.datasets = OrderedDict()
        self.datasets["SegPath"] = 1_000 # 25_674
        self.datasets["SegPath_All"] = 1_000 # 584_796
        self.datasets["SNOW"] = 1_000 # 20_000
        self.datasets["NeurIPS"] = 1_000 # 1_679
        self.datasets["TissueNet"] = 1_000 # 7_022
        self.datasets["DynamicNuclearNet"] = 1_000 # 7_084
        self.datasets["BCData"] = 1_000 # 1_338
        self.datasets["PanNuke"] = 1_000 # 7_901
        self.datasets["DSB2018"] = 1_000 # 37_333
        self.datasets["BBBC"] = 1_000 # 11_423_738 # ~/Projects/Stainalyzer/supplementary_data/images_to_enhance
        self.datasets["DAB"] = 4_977

        # Metrics
        self.metrics = OrderedDict()
        self.metrics["Segm"] = ["F1SEG", "IoU", "PxAcc", "AJI", "HD95", "F1BDR", "F1DET", "PQ", "ARI", "RECSEG", "PRESEG"]
        self.metrics["Stai"] = ["MOD", "IOD", "HEDAB", "CDE", "CVAR", "SNE", "ESD", "BSI", "CTNR"]
        self.metrics["Hard"] = ["IT", "TPT", "VRAM", "RAM", "MACs", "BW", "EPI", "TTA", "LAT", "MTTF"]
        self.metrics["Clas"] = ["CMA", "F1S", "FBET", "BSCO", "CROE", "COHK", "SPE", "AUROC", "AUPR", "MCC"]
        self.metrics["Allt"] = ["CWAV", "SWAV", "HWAV", "LWAV", "CSWA", "CSHWA", "CSHLWA"]

    def create_segm_tables(self, output_path : Path = None):
        """
        Database Names: [NB]_[DATABASE]_Segm_[METRIC]
        Database Shapes: Images x Methods
        ----------------------------------------------------
        Example: 1_SegPath_Segm_F1SEG.tsv
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
            for metric in self.metrics["Segm"]:

                # Image IDs (assuming naming scheme)
                image_ids = [f"Image{str(i).zfill(self.number_of_digits(n_images))}" for i in range(n_images)]

                # Method name -> vector (length = n_images)
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
                filename = f"{dataset_number}_{dataset}_Segm_{metric}.tsv"
                out_path = output_path / filename

                # Save as TSV
                out_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_path, sep="\t", index=True)

                break
            break

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

    def create_class_tables(self):
        """
        Database Names: [NB]_[DATABASE]_Clas_[METRIC]
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

    def create_allt_tables(self):
        """
        Database Names: [NB]_[DATABASE]_AllT_[METRIC]
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
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1) # Check minV and maxV
        elif metric == "IoU":
            f2 = self._create_2_iou_cell
            v = list(f2(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "PxAcc":
            f3 = self._create_3_pxacc_cell
            v = list(f3(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "AJI": 
            f4 = self._create_4_aji_cell
            v = list(f4(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "HD95":
            f5 = self._create_5_hd95_cell
            v = list(f5(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "F1BDR":
            f6 = self._create_6_f1bdr_cell
            v = list(f6(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "F1DET": 
            f7 = self._create_7_f1det_cell
            v = list(f7(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "PQ":
            f8 = self._create_8_pq_cell
            v = list(f8(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "ARI":
            f9 = self._create_9_ari_cell
            v = list(f9(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "RECSEG":
            f10 = self._create_10_recseg_cell
            v = list(f10(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "PRESEG":
            f11 = self._create_11_preseg_cell
            v = list(f11(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
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

    def _create_4_aji_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_4_aji_cell
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

        return {"aji_cell": vector}

    def _create_5_hd95_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_5_hd95_cell
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

    def _create_6_f1bdr_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_6_f1bdr_cell
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

    def _create_7_f1det_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_7_f1det_cell
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

        return {"f1det_cell": vector}

    def _create_8_pq_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_8_pq_cell
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

    def _create_9_ari_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_9_ari_cell
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

    def _create_10_recseg_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_10_recseg_cell
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

    def _create_11_preseg_cell(self, n_images, method_rank=1, function=None, *args, **kwargs):
        """
        _create_11_preseg_cell
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

        return {"f1det_nucleus": vector}

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

    # Classification
    def _call_clas_function(self,
                            metric : str, 
                            n_images : int, 
                            method_rank : int = 1, 
                            function : Callable = None, 
                            *args : Optional[List[Any]], 
                            **kwargs : Optional[Dict[str, Any]]):
        """
        _call_clas_function
        """

        # New value(s)
        v = 0.0
        glitchn = np.random.randint(5, 15)

        # Function Routing
        if metric == "CMA" or metric == "F1S" or metric == "SPE":
            f1 = self._create_1_cma_clas 
            v = list(f1(n_images, method_rank=method_rank, function=function, *args, **kwargs).values()) #[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "FBET":
            f2 = self._create_2_fbet_clas
            v = list(f2(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "BSCO":
            f3 = self._create_3_bsco_clas
            v = list(f3(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "CROE":
            f4 = self._create_4_croe_clas
            v = list(f4(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "COHK":
            f5 = self._create_5_cohk_clas
            v = list(f5(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "AUROC":
            f6 = self._create_6_auroc_clas
            v = list(f6(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "AUPR":
            f7 = self._create_7_aupr_clas
            v = list(f7(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        elif metric == "MCC":
            f7 = self._create_8_mcc_clas
            v = list(f8(n_images, method_rank=method_rank, function=function, *args, **kwargs).values())[0][0]
            v = self._add_glitch(v, glitchn, maxv=1.0, minv=0.0, mult=1)
        else:
            raise ValueError("Invalid function option.")

        return v

    # Confusion Matrix + F1-score + Specificity
    def _create_1_cma_clas(self,
                           n_images: int, 
                           method_rank: int = 1, 
                           function: Callable = None, 
                           *args: List[Any], 
                           **kwargs: Dict[Any, Any]) -> Dict[str, List[Union[int, float]]]:
        """
        Simulate a confusion matrix and derived classification metrics for mock statistical evaluation.
        Returns a list of [TP, FP, TN, FN, F1, Precision, Recall, Specificity].
        """

        # Initialize counts
        tp, fp, tn, fn = 0, 0, 0, 0

        # Method-dependent generation
        if method_rank == 1:
            tp = beta.rvs(a=10, b=1, size=1)[0] * 0.4 * n_images
            tn = beta.rvs(a=10, b=1, size=1)[0] * 0.5 * n_images
            fp = beta.rvs(a=1, b=8, size=1)[0] * 0.05 * n_images
            fn = beta.rvs(a=1, b=8, size=1)[0] * 0.05 * n_images
        elif method_rank == 2:
            tp = beta.rvs(a=8, b=2, size=1)[0] * 0.35 * n_images
            tn = beta.rvs(a=8, b=2, size=1)[0] * 0.45 * n_images
            fp = beta.rvs(a=2, b=6, size=1)[0] * 0.1 * n_images
            fn = beta.rvs(a=2, b=6, size=1)[0] * 0.1 * n_images
        elif method_rank == 3:
            tp = beta.rvs(a=5, b=5, size=1)[0] * 0.3 * n_images
            tn = beta.rvs(a=5, b=5, size=1)[0] * 0.4 * n_images
            fp = beta.rvs(a=3, b=4, size=1)[0] * 0.15 * n_images
            fn = beta.rvs(a=3, b=4, size=1)[0] * 0.15 * n_images
        elif method_rank == 4:
            tp = beta.rvs(a=3, b=6, size=1)[0] * 0.25 * n_images
            tn = beta.rvs(a=4, b=5, size=1)[0] * 0.35 * n_images
            fp = beta.rvs(a=4, b=3, size=1)[0] * 0.2 * n_images
            fn = beta.rvs(a=4, b=3, size=1)[0] * 0.2 * n_images
        elif method_rank == 5:
            tp = beta.rvs(a=2, b=8, size=1)[0] * 0.2 * n_images
            tn = beta.rvs(a=3, b=6, size=1)[0] * 0.3 * n_images
            fp = beta.rvs(a=5, b=2, size=1)[0] * 0.25 * n_images
            fn = beta.rvs(a=5, b=2, size=1)[0] * 0.25 * n_images
        else:
            tp = beta.rvs(a=1, b=10, size=1)[0] * 0.15 * n_images
            tn = beta.rvs(a=2, b=8, size=1)[0] * 0.25 * n_images
            fp = beta.rvs(a=6, b=1, size=1)[0] * 0.3 * n_images
            fn = beta.rvs(a=6, b=1, size=1)[0] * 0.3 * n_images

        # Convert to integers
        tp, fp, tn, fn = map(int, [tp, fp, tn, fn])

        # Adjust in case the total is out of bounds
        total = tp + fp + tn + fn
        if total > n_images:
            scale = n_images / total
            tp = int(tp * scale)
            fp = int(fp * scale)
            tn = int(tn * scale)
            fn = int(fn * scale)

        # Derived metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        return {
            "cma_clas": [tp, fp, tn, fn, round(f1_score, 4), round(precision, 4), round(recall, 4), round(specificity, 4)]
        }

    # Fβ-score
    def _create_2_fbet_clas(self,
                            n_images: int,
                            method_rank: int = 1,
                            function: Callable = None,
                            *args: List[Any],
                            **kwargs: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Simulates F-beta score values for classification evaluation using various statistical distributions.
        """
        if method_rank == 1:
            vector = beta.rvs(a=15, b=2, size=n_images)
        elif method_rank == 2:
            vector = beta.rvs(a=10, b=3, size=n_images)
        elif method_rank == 3:
            vector = norm.rvs(loc=0.8, scale=0.07, size=n_images).clip(0, 1)
        elif method_rank == 4:
            vector = triang.rvs(c=0.4, loc=0.4, scale=0.4, size=n_images).clip(0, 1)
        elif method_rank == 5:
            vector = lognorm.rvs(s=0.4, scale=np.exp(0.3), size=n_images).clip(0, 1)
        elif method_rank == 6:
            vector = gompertz.rvs(c=1.5, size=n_images).clip(0, 1)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = norm.rvs(loc=0.45, scale=0.15, size=n_images).clip(0, 1)

        return {"fbet_clas": list(vector)}

    # Brier score
    def _create_3_bsco_clas(self,
                            n_images: int,
                            method_rank: int = 1,
                            function: Callable = None,
                            *args: List[Any],
                            **kwargs: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Simulates Brier Score values for classification evaluation.
        Lower is better. Range: [0, 1]
        """
        if method_rank == 1:
            vector = beta.rvs(a=1, b=15, size=n_images)  # strongly skewed to low values
        elif method_rank == 2:
            vector = beta.rvs(a=2, b=10, size=n_images)
        elif method_rank == 3:
            vector = norm.rvs(loc=0.25, scale=0.08, size=n_images).clip(0, 1)
        elif method_rank == 4:
            vector = triang.rvs(c=0.3, loc=0.2, scale=0.4, size=n_images).clip(0, 1)
        elif method_rank == 5:
            vector = lognorm.rvs(s=0.6, scale=np.exp(-0.3), size=n_images).clip(0, 1)
        elif method_rank == 6:
            vector = expon.rvs(scale=0.5, size=n_images).clip(0, 1)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = norm.rvs(loc=0.5, scale=0.2, size=n_images).clip(0, 1)

        return {"bsco_clas": list(vector)}

    # Normalized Cross-Entropy
    def _create_4_croe_clas(self,
                            n_images: int,
                            method_rank: int = 1,
                            function: Callable = None,
                            *args: List[Any],
                            **kwargs: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Simulates Normalized Cross-Entropy values for classification evaluation.
        Lower is better. Range: [0, 1]
        """
        if method_rank == 1:
            vector = beta.rvs(a=1.5, b=20, size=n_images)
        elif method_rank == 2:
            vector = beta.rvs(a=2, b=10, size=n_images)
        elif method_rank == 3:
            vector = norm.rvs(loc=0.3, scale=0.08, size=n_images).clip(0, 1)
        elif method_rank == 4:
            vector = triang.rvs(c=0.5, loc=0.2, scale=0.5, size=n_images).clip(0, 1)
        elif method_rank == 5:
            vector = gompertz.rvs(c=1.5, size=n_images).clip(0, 1)
        elif method_rank == 6:
            vector = expon.rvs(scale=0.6, size=n_images).clip(0, 1)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = norm.rvs(loc=0.5, scale=0.2, size=n_images).clip(0, 1)

        return {"croe_clas": list(vector)}

    # Cohen's Kappa
    def _create_5_cohk_clas(self,
                            n_images: int,
                            method_rank: int = 1,
                            function: Callable = None,
                            *args: List[Any],
                            **kwargs: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Simulates Cohen's Kappa values for classification evaluation.
        Higher is better. Range: [-1, 1], but typically in [0, 1]
        """
        if method_rank == 1:
            vector = beta.rvs(a=10, b=1.5, size=n_images)
        elif method_rank == 2:
            vector = beta.rvs(a=8, b=2, size=n_images)
        elif method_rank == 3:
            vector = norm.rvs(loc=0.7, scale=0.1, size=n_images).clip(0, 1)
        elif method_rank == 4:
            vector = triang.rvs(c=0.3, loc=0.2, scale=0.6, size=n_images).clip(0, 1)
        elif method_rank == 5:
            vector = gompertz.rvs(c=1.0, size=n_images).clip(0, 0.6)
        elif method_rank == 6:
            vector = expon.rvs(scale=0.2, size=n_images).clip(0, 0.4)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = norm.rvs(loc=0.4, scale=0.2, size=n_images).clip(-1, 1)

        return {"cohk_clas": list(vector)}

    # Area under the ROC Curve
    def _create_6_auroc_clas(self,
                             n_images: int,
                             method_rank: int = 1,
                             function: Callable = None,
                             *args: List[Any],
                             **kwargs: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Simulates Area Under the ROC Curve values for classification evaluation.
        Higher is better. Range: [0, 1]
        """
        if method_rank == 1:
            vector = beta.rvs(a=12, b=2, size=n_images)
        elif method_rank == 2:
            vector = beta.rvs(a=10, b=3, size=n_images)
        elif method_rank == 3:
            vector = norm.rvs(loc=0.8, scale=0.07, size=n_images).clip(0, 1)
        elif method_rank == 4:
            vector = triang.rvs(c=0.4, loc=0.5, scale=0.3, size=n_images).clip(0, 1)
        elif method_rank == 5:
            vector = gompertz.rvs(c=1.2, size=n_images).clip(0.4, 0.75)
        elif method_rank == 6:
            vector = expon.rvs(scale=0.4, size=n_images).clip(0.3, 0.7)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = norm.rvs(loc=0.5, scale=0.2, size=n_images).clip(0, 1)

        return {"auroc_clas": list(vector)}

    # Area Under the Precision Recall Curve
    def _create_7_aupr_clas(self,
                            n_images: int,
                            method_rank: int = 1,
                            function: Callable = None,
                            *args: List[Any],
                            **kwargs: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Simulates Area Under the Precision-Recall Curve (AUPRC) values.
        Higher is better. Range: [0, 1]
        """
        if method_rank == 1:
            vector = beta.rvs(a=14, b=2, size=n_images)
        elif method_rank == 2:
            vector = beta.rvs(a=10, b=3, size=n_images)
        elif method_rank == 3:
            vector = norm.rvs(loc=0.78, scale=0.08, size=n_images).clip(0, 1)
        elif method_rank == 4:
            vector = triang.rvs(c=0.4, loc=0.4, scale=0.4, size=n_images).clip(0, 1)
        elif method_rank == 5:
            vector = gompertz.rvs(c=1.3, size=n_images).clip(0.3, 0.75)
        elif method_rank == 6:
            vector = expon.rvs(scale=0.45, size=n_images).clip(0.2, 0.6)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = norm.rvs(loc=0.5, scale=0.2, size=n_images).clip(0, 1)

        return {"aupr_clas": list(vector)}

    # Matthews Correlation Coefficient
    def _create_8_mcc_clas(self,
                           n_images: int,
                           method_rank: int = 1,
                           function: Callable = None,
                           *args: List[Any],
                           **kwargs: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Simulates Matthews Correlation Coefficient (MCC) values for classification.
        Higher is better. Range: [-1, 1], but typically in [0, 1].
        """
        if method_rank == 1:
            vector = beta.rvs(a=12, b=2, size=n_images)
        elif method_rank == 2:
            vector = beta.rvs(a=10, b=3, size=n_images)
        elif method_rank == 3:
            vector = norm.rvs(loc=0.75, scale=0.1, size=n_images).clip(0, 1)
        elif method_rank == 4:
            vector = triang.rvs(c=0.3, loc=0.3, scale=0.5, size=n_images).clip(0, 1)
        elif method_rank == 5:
            vector = gompertz.rvs(c=1.4, size=n_images).clip(0, 0.6)
        elif method_rank == 6:
            base = expon.rvs(scale=0.4, size=n_images).clip(0, 0.5)
            noise = norm.rvs(loc=-0.1, scale=0.1, size=n_images)
            vector = (base + noise).clip(-1, 0.5)
        else:
            if function is not None:
                vector = function(*args, **kwargs)
            else:
                vector = norm.rvs(loc=0.5, scale=0.2, size=n_images).clip(-1, 1)

        return {"mcc_clas": list(vector)}

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    # DAB-Staining
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

    ### Auxiliary methods:

    def number_of_digits(self, n):
        """
        Return the number of digits of n
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        return int(math.log10(n)) + 1

    ### Object Fixing:

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
        proportions = Proportions()
        final_dict = proportions.proportions_dict

        # Iterating on proportions and methods dictionaries
        for method_name, method_rank in methods_dict.items():
            prop_vec = proportions_dict[method_rank]
            final_dict[method_name] = self._appendage(n_images, prop_vec)

        # Return real rank dictionary
        return final_dict

    ### Glitching:

    def _add_glitch(self, value, n, maxv=1.0, minv=0.0, mult=1) -> np.float64:
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

        return np.float64(max(min(value, maxv), minv))

    def _glitch(self, value, type="pico", function=None, *args, **kwargs) -> float:
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

