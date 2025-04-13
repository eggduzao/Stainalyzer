"""
visualization
-------------

# Example Usage
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

import io
import os
import cv2
import math
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import scipy.stats as stats
from typing import Optional
from openpyxl import Workbook
from PIL import Image, ImageCms
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.special import expit, softmax

from .utils import ColorName, ColorConverter, PlottingUtils

###############################################################################
# Constants
###############################################################################

# Constants
SEED = 1987
np.random.seed(SEED)

###############################################################################
# Classes
###############################################################################

class VizPlots:
    """
    A comprehensive visualization class for generating various plots based on quality metrics.

    Methods
    -------
    pie_chart(data, labels, is_percent=False, plot_as_percent=False)
        Generates a pie chart with customizable formatting.
    star_plot(data)
        Generates a star plot with values from 1 to 5.
    bar_plot(data, x_labels, y_labels, x_title, y_title, inverted=False)
        Generates a bar plot with customizable aesthetics.
    box_plot(data)
        Generates a violin plot to represent distributions.
    distribution_plot(data)
        Generates overlapping distribution plots.
    polar_plot(data)
        Generates a polar plot.
    """

    def __init__(self):
        pass

    def heatmap(self,
                data: pd.DataFrame,
                normalize: bool = False,
                bin_rows: Optional[int] = None,
                title: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                xrotation: int = 45,
                yrotation: int = 0,
                output_file_name: Optional[Path] = None,
                ) -> None:
        """
        Generates a beautiful heatmap from a DataFrame, optionally reducing rows via binning.

        Parameters:
        - data (pd.DataFrame): The input data, should have method columns and image index.
        - normalize (bool): If True, normalize values column-wise between 0 and 1.
        - title, xlabel, ylabel: Optional plot labels.
        - xrotation, yrotation: Rotation angles for x and y axis labels.
        - output_file_name (Path): If set, saves to file instead of showing.
        - bin_rows (int): If set, reduce number of rows by this number of bins.


        Example
        --------
        beautiful_heatmap(data=df,
                          normalize=True,
                          title="Segmentation Performance by Method",
                          xlabel="Segmentation Methods",
                          ylabel="Image Clusters",
                          xrotation=45,
                          yrotation=0,
                          bin_rows=50,  # ← Smart compression here!
                          output_file_name=None  # Or Path("heatmap_output.tiff")
                          )
        """

        # Copy dataframe for normalization and reduction control
        df = data.copy()

        # Reduction of very large datasets
        if bin_rows is not None:
            df = self._reduce_dataframe_by_bins(df, n_bins=bin_rows)

        # Normalize (column-wise min-max scaling)
        if normalize:
            df = df.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
        
        # Create a figure
        plt.figure(figsize=(12, 14))
        
        # Choose a beautiful colormap
        cmap = sns.diverging_palette(220, 20, as_cmap=True) # l=65, center="light",
        
        # Plot heatmap
        ax = sns.heatmap(df, annot=False, cmap=cmap, linewidths=0.5, linecolor="gray")
        
        # Titles and labels
        if title is not None:
            plt.title(title, fontsize=16, fontweight='bold', pad=15)
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=14)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=14)
        
        # Improve spacing
        plt.xticks(rotation=xrotation, ha="right", fontsize=10)
        plt.yticks(rotation=yrotation, fontsize=8)
        
        # Show the plot
        plt.tight_layout()
        if output_file_name is not None:
            plt.savefig(output_file_name, dpi=300, format="tiff", bbox_inches='tight')
            return
        plt.show()

    def _reduce_dataframe_by_bins(self, df: pd.DataFrame, n_bins: int = 50) -> pd.DataFrame:
        """
        Reduces a DataFrame by binning the rows based on equal-sized bins over the index
        and taking the mean per bin. Assumes all columns are numeric and aligned across methods.

        Parameters:
        - df (pd.DataFrame): Input DataFrame with numeric values.
        - n_bins (int): Number of bins to divide the data into.

        Returns:
        - pd.DataFrame: Reduced DataFrame with one row per bin and averaged values.
        """

        if df.shape[0] <= n_bins:
            return df.copy()

        # Sort by mean across methods to ensure fair binning along the spectrum
        df_sorted = df.copy()
        df_sorted["__mean__"] = df.mean(axis=1)
        df_sorted.sort_values("__mean__", inplace=True)
        df_sorted.drop(columns="__mean__", inplace=True)

        # Assign each row to a bin
        df_sorted["__bin__"] = pd.qcut(np.arange(len(df_sorted)), q=n_bins, labels=False)

        # Group by bin and average
        reduced_df = df_sorted.groupby("__bin__").mean()

        return reduced_df

    def violin_plot(self,
                    data : pd.DataFrame,
                    normalize : bool = False, 
                    title : str = None,
                    xlabel : str = None,
                    ylabel : str = None,
                    alpha : float = 0.8, 
                    palette : str = "colorblind", 
                    log_scale : bool = False,
                    xrotation : int = 45,
                    yrotation : int = 0,
                    output_file_name : Path() = None) -> None:
        """
        # colorblind    Seaborn’s default colorblind-safe palette
        # deep          Well-separated colors, readable in B&W
        # muted         Softer colors, but still distinct
        # dark          High-contrast for B&W printing
        # Set1          Bold and distinguishable (also works in B&W)
        # Set2          Softer, still good for B&W
        # Set3          Larger variety of distinguishable colors
        # Paired        Good for categories with similar colors
        # cubehelix     Monochrome-variant, useful for B&W
        # husl          Hue-based, optimized for color vision issues

        # Save in different formats
        #fig.savefig("figure.png", dpi=300, format="png")  # High DPI PNG
        #fig.savefig("figure.tiff", dpi=300, format="tiff")  # TIFF (print-friendly)
        #fig.savefig("figure.eps", format="eps", dpi=300)  # EPS (vector format, no transparency)
        #fig.savefig("figure.pdf", format="pdf", transparent=True, dpi=300)  # PDF (editable)
        #fig.savefig("figure.svg", format="svg", transparent=True, dpi=300)  # SVG (best for editing)
        """
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))  # Set figure size (width, height in inches)

        # Create the violin plot with thicker boxplots and a white background outlined in black
        ax = sns.violinplot(
            data=df,
            inner="box",  # Ensures inner boxplots are drawn
            linewidth=1.0,  # Makes boxplot outlines thicker
            palette=palette, 
            alpha=alpha
        )

        if log_scale:
            ax.set_yscale("log")  # Use logarithmic scale for Y-axis
        
        # Add black dots for the median of each violin plot
        for i, col in enumerate(df.columns):
            median_value = df[col].median()
            plt.scatter(i, median_value, color=sns.color_palette(palette)[i], edgecolors="black", s=50, zorder=3)

        # Adjust boxplot elements for better visibility
        for artist in ax.artists:
            artist.set_edgecolor("black")  # Outline the violins in black
            artist.set_facecolor("white")  # Make the boxplots have a white background
            artist.set_linewidth(1.5)  # Thicker lines

        # Add outliers as black dots
        sns.boxplot(
        data=df,
        width=0.3,  # <-- Reduce this to make boxplots thinner (default ~0.6)
        showfliers=True,  
        fliersize=5,  
        linewidth=1.5,  
        boxprops={"facecolor": "none"},  
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 0},  # <-- Set linewidth to 0 to REMOVE whisker caps
        medianprops={"linewidth": 2},
        flierprops={"marker": "o", "color": "black", "markersize": 5}
        )

        # Add horizontal line
        plt.axhline(y=0, color="red", linestyle="dashed", linewidth=1)  # Red dashed line at y=0
        plt.axhline(y=1, color="blue", linestyle="dotted", linewidth=1)  # Blue dotted line at y=1

        # Add vertical line
        #plt.axvline(x=2, color="green", linestyle="dashed", linewidth=1)  # Green dashed line at x=2
        #plt.axvline(x=4, color="purple", linestyle="dotted", linewidth=1)  # Purple dotted line at x=4

        # Remove axis details
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_frame_on(False)
        if output_file_name is not None:
            plt.savefig(output_file_name, dpi=300, format="tiff")
            return
        plt.show()

    def pie_chart(self, values=None, labels=None, data=None, is_percent=False, plot_as_percent=False):
        """
        Generates a pie chart with optional data transformation to percentages.

        Parameters
        ----------
        values : list, optional
            List of numerical values.
        labels : list, optional
            List of labels for the pie chart.
        data : dict, optional
            Dictionary with labels as keys and numerical values as values.
        is_percent : bool, optional
            If True, data is already in percentage.
        plot_as_percent : bool, optional
            If True, display percentages on the pie chart.
        """
        if data:
            labels, values = zip(*data.items())

        if not is_percent:
            total = sum(values)
            values = [v / total * 100 for v in values]

        colors = sns.color_palette('pastel')
        explode = [0.05] * len(values)

        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%' if plot_as_percent else None,
            startangle=140, colors=colors, explode=explode, shadow=True
        )
        plt.title("Pie Chart")
        plt.show()

    def star_plot(self, data):
        """
        Generates a star plot for 1-to-5 ratings.

        Parameters
        ----------
        data : dict
            Dictionary with categories as keys and integer ratings (1-5) as values.
        """
        labels, values = zip(*data.items())

        fig, ax = plt.subplots(figsize=(6, 6))
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
        values += (values[0],)
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        ax.fill(angles, values, color='lightblue', alpha=0.6)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.title("Star Plot")
        plt.show()

    def bar_plot(self, data, x_labels=None, y_labels=None, x_title="", y_title="", inverted=False):
        """
        Generates a bar plot with customizable options.

        Parameters
        ----------
        data : list
            List of numerical values or list of lists for series.
        x_labels : list, optional
            Labels for the x-axis.
        y_labels : list, optional
            Labels for the y-axis.
        x_title : str, optional
            Title for the x-axis.
        y_title : str, optional
            Title for the y-axis.
        inverted : bool, optional
            If True, inverts the axes.
        """
        if isinstance(data[0], (list, np.ndarray)):
            df = pd.DataFrame(data).T
            df.plot(kind="bar", stacked=True)
        else:
            plt.bar(x_labels, data)

        plt.xlabel(x_title)
        plt.ylabel(y_title)
        if inverted:
            plt.gca().invert_yaxis()
        plt.title("Bar Plot")
        plt.show()

    def box_plot(self, data, violin=True):
        """
        Generates box or violin plots for distributions.

        Parameters
        ----------
        data : list of lists
            List of distributions.
        violin : bool, optional
            If True, use violin plot instead of box plot.
        """
        df = pd.DataFrame(data).T
        if violin:
            sns.violinplot(data=df, palette='pastel')
        else:
            sns.boxplot(data=df, palette='pastel')
        plt.title("Box/Violin Plot")
        plt.show()

    def distribution_plot(self, data):
        """
        Generates distribution plots with overlapping histograms.

        Parameters
        ----------
        data : list of lists
            List of distributions.
        """
        df = pd.DataFrame(data).T
        for col in df.columns:
            sns.kdeplot(df[col], fill=True)
        plt.title("Distribution Plot")
        plt.show()

    def polar_plot(self, data):
        """
        Generates a polar plot for metrics.

        Parameters
        ----------
        data : dict
            Dictionary with metric names as keys and numerical values as values.
        """
        labels, values = zip(*data.items())
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
        values += (values[0],)
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        ax.fill(angles, values, color='lightgreen', alpha=0.6)
        ax.plot(angles, values, color='green', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.title("Polar Plot")
        plt.show()

