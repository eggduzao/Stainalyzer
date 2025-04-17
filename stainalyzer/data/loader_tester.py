
"""
Tester
------

Placeholder.
"""

############################################################################################################
### Import
############################################################################################################

import io
import sys
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from time import perf_counter
from PIL import Image, ImageCms
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from collections import OrderedDict

from stainalyzer.data.loader import MetricTables
from stainalyzer.data.loader_visualization import VizPlots

############################################################################################################
### Classes
############################################################################################################

class LoaderTester:
    """
    A class to manage the evaluation and testing process for trained distribution models.

    The Tester class is responsible for:
    - Loading and preprocessing test data.
    - Evaluating the performance of trained models on test datasets.
    - Comparing trained models using various metrics.
    - Generating detailed evaluation reports.
    - Logging progress and results for transparency.

    Attributes:
        model (object): The trained model to be tested (e.g., DABDistribution, MultivariateLABDistribution).
        test_data (list or numpy.ndarray): The dataset used for testing.
        metrics (dict): Dictionary of metrics to evaluate model performance.
        logger (object, optional): Logger for tracking testing progress and results.
    """

    def __init__(self,
                 input_path : Path = None,
                 output_path : Path = None):
        """
        Initialize the Tester class.

        Parameters:
            model (object, optional): The trained model to be tested. Default is None.
            test_data (list or numpy.ndarray, optional): Dataset for testing. Default is None.
            metrics (dict, optional): Evaluation metrics to use. Default is None.
            logger (object, optional): Logger for tracking progress. Default is None.
        """
        self.input_path = input_path if input_path.exists() else None
        self.output_path = output_path if output_path.exists() else None

    def main(self):
        """
        Placeholder

        Returns:
            Placeholder: Placeholder.

        Raises:
            Placeholder: Placeholder.
        """

        # Tables
        results_path = Path("/Users/egg/Projects/Stainalyzer/data/results/")
        tester = Tester(results_path)
        #tester.execute()

        # Calculate "Cell" tables
        cell_path = self.model_path / "1_cell"
        cell_path.mkdir(parents=True, exist_ok=True)
        metric_tables = MetricTables()
        metric_tables.create_cell_tables(output_path=cell_path)

        # Plots
        table_path = results_path / "1_cell" / "1_SegPath_Cell_F1SEG.tsv"
        output_path = results_path / "1_cell" / "1_SegPath_Cell_F1SEG.tiff"
        tester.plotter(table_path, output_path)

    def plotter(self, input_file_name : Path, output_file_name : Path = None):
        """
        # df = pd.read_csv("your_file.tsv", sep="\t", header=None)
        # df = pd.read_csv("your_file.tsv", sep="\t", encoding="utf-8")
        """

        # Plotting Test
        v = VizPlots()
        data_frame = pd.read_csv(input_file_name, sep = "\t", index_col=0)
        v.heatmap(data_frame, normalize=True, bin_rows=50, output_file_name = output_file_name)

    def _generate(self):
        """ Generator """
        pass
        # numpy.random ----------------------------------------------------------------------------
        # Distribution    Function    Parameters
        # Uniform numpy.random.uniform(low, high, size=N) low (min), high (max)
        # Normal (Gaussian)   numpy.random.normal(loc, scale, size=N) loc (mean), scale (std dev)
        # Log-Normal  numpy.random.lognormal(mean, sigma, size=N) mean, sigma
        # Exponential numpy.random.exponential(scale, size=N) scale (1/λ)
        # Gamma   numpy.random.gamma(shape, scale, size=N)    shape, scale
        # Beta    numpy.random.beta(a, b, size=N) a, b
        # Chi-Square  numpy.random.chisquare(df, size=N)  df (degrees of freedom)
        # Dirichlet   numpy.random.dirichlet(alpha, size=N)   alpha (parameter vector)
        # Poisson numpy.random.poisson(lam, size=N)   lam (λ, mean number of events)
        # Binomial    numpy.random.binomial(n, p, size=N) n (trials), p (success prob)
        # Geometric   numpy.random.geometric(p, size=N)   p (success probability)

        # scipy.stats -----------------------------------------------------------------------------
        # Distribution    Function    Parameters
        # Uniform scipy.stats.uniform.rvs(loc, scale, size=N) loc, scale
        # Normal (Gaussian)   scipy.stats.norm.rvs(loc, scale, size=N)    loc, scale
        # Log-Normal  scipy.stats.lognorm.rvs(sigma, loc, scale, size=N)  sigma, loc, scale
        # Exponential scipy.stats.expon.rvs(scale, size=N)    scale (1/λ)
        # Gamma   scipy.stats.gamma.rvs(shape, scale, size=N) shape, scale
        # Beta    scipy.stats.beta.rvs(a, b, size=N)  a, b
        # Chi-Square  scipy.stats.chi2.rvs(df, size=N)    df (degrees of freedom)
        # Dirichlet   scipy.stats.dirichlet.rvs(alpha, size=N)    alpha (parameter vector)
        # Poisson scipy.stats.poisson.rvs(mu, size=N) mu (mean λ)
        # Binomial    scipy.stats.binom.rvs(n, p, size=N) n, p
        # Negative Binomial   scipy.stats.nbinom.rvs(n, p, size=N)    n, p
        # Geometric   scipy.stats.geom.rvs(p, size=N) p
        # Student’s t scipy.stats.t.rvs(df, size=N)   df
        # Weibull scipy.stats.weibull_min.rvs(c, size=N)  c (shape parameter)
        # Cauchy  scipy.stats.cauchy.rvs(loc, scale, size=N)  loc, scale
