"""
====================================================================================================
loader_visualizations.py - placeholder
====================================================================================================

Overview
--------
This module, 'loader_visualizations.py', placeholder.


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

# Register Consolas
# consolas_path = "/Library/Fonts/Consolas.ttf"
# consolas_font = fm.FontProperties(fname=consolas_path)
# plt.rcParams["font.family"] = consolas_font.get_name()

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

    """
    def violin_plot(self,
                    *arrays: np.ndarray,
                    array_names: List,
                    output_path: Path,
                    means: np.ndarray = None,
                    stds: np.ndarray = None,
                    normalize: bool = False,
                    title: str = None,
                    xlabel: str = None,
                    ylabel: str = None,
                    alpha: float = 0.9,
                    palette: str = "colorblind",
                    log_scale: bool = False,
                    xrotation: int = 0,
                    yrotation: int = 0) -> None:
        
        Create a beautiful violin plot from one or more NumPy arrays and save it to disk.

        Parameters
        ----------
        *arrays : np.ndarray
            One or more 1D NumPy arrays to plot as violins.
        output_path : Path
            Path object specifying where to save the resulting plot.
        means : np.ndarray, optional
            Precomputed mean values to display above each violin. If None, they are computed automatically.
        stds : np.ndarray, optional
            Precomputed standard deviation values to display above each violin. If None, they are computed automatically.
        normalize : bool, default=False
            Whether to normalize each array before plotting.
        title : str, optional
            Title of the plot.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        alpha : float, default=0.9
            Transparency level for violins.
        palette : str, default='colorblind'
            Seaborn color palette to use.
        log_scale : bool, default=False
            Apply logarithmic scale to Y-axis.
        xrotation : int, default=0
            Rotation angle for x-axis labels.
        yrotation : int, default=0
            Rotation angle for y-axis labels.

        Pallete Examples
        ----------------
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

        Saving Examples
        ---------------
        #fig.savefig("figure.png", dpi=300, format="png")  # High DPI PNG
        #fig.savefig("figure.tiff", dpi=300, format="tiff")  # TIFF (print-friendly)
        #fig.savefig("figure.eps", format="eps", dpi=300)  # EPS (vector format, no transparency)
        #fig.savefig("figure.pdf", format="pdf", transparent=True, dpi=300)  # PDF (editable)
        #fig.savefig("figure.svg", format="svg", transparent=True, dpi=300)  # SVG (best for editing)

        Returns
        -------
        None
            Saves the plot to the specified output_path.
        
        # Prepare the data
        df = pd.DataFrame({
            name: arr / np.linalg.norm(arr) if normalize else arr
            for name, arr in zip(array_names, arrays)
        })

        # Start plot
        fig, ax = plt.subplots(figsize=(max(6, len(arrays) * 1.2), 5))

        # Set font to Consolas
        # plt.rcParams["font.family"] = "Consolas"

        # Violin plot
        sns.violinplot(
            data=df,
            inner="box",
            linewidth=1.2,
            palette=palette,
            alpha=alpha,
            ax=ax
        )

        # Add standard deviation and mean text
        for i, column in enumerate(df.columns):
            mean = np.mean(df[column]) if means is None else means[i]
            std = np.std(df[column], ddof=1) if stds is None else stds[i]
            ax.text(
                i,
                df[column].max() * 1.05,
                f"mean = {mean:.4f}\nstd = {std:.4f}",
                ha="center",
                va="bottom",
                fontsize=9
                # fontfamily="Consolas"
            )

        # Axis and labels
        if title: ax.set_title(title, fontsize=12)
        if xlabel: ax.set_xlabel(xlabel, fontsize=10)
        if ylabel: ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='x', rotation=xrotation)
        ax.tick_params(axis='y', rotation=yrotation)
        if log_scale:
            ax.set_yscale('log')

        # Clean up and save
        sns.despine()
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, format=output_path.suffix[1:])
        plt.close()
    """

    def violin_plot(self,
                    arrays: List[np.ndarray],
                    array_names: List[str],
                    output_path: Path,
                    means: List[float] = None,
                    stds: List[float] = None,
                    normalize: bool = False,
                    title: str = None,
                    xlabel: str = None,
                    ylabel: str = None,
                    alpha: float = 0.8,
                    palette: str = "colorblind",
                    log_scale: bool = False,
                    xrotation: int = 0,
                    yrotation: int = 0
                    ) -> None:
        """
        Plot beautiful violin plots with optional normalization, mean/std text, 
        and pairwise variance significance annotations using an F-test.

        Parameters
        ----------
        arrays : List[np.ndarray]
            List of 1D numpy arrays to be plotted.
        array_names : List[str]
            Labels for each violin (must match number of arrays).
        output_path : Path
            Path to save the generated figure.
        normalize : bool, optional
            Whether to normalize each array using L2 norm (default: False).
        title : str, optional
            Plot title.
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.
        alpha : float, optional
            Transparency level for violins (default: 0.8).
        palette : str, optional
            Seaborn color palette to use (default: "colorblind").
        log_scale : bool, optional
            Use log scale for y-axis (default: False).
        xrotation : int, optional
            Rotation of x-axis ticks.
        yrotation : int, optional
            Rotation of y-axis ticks.
        means : List[float], optional
            Optional list of means to display.
        stds : List[float], optional
            Optional list of standard deviations to display.
        """

        # Prepare DataFrame
        df = pd.DataFrame({
            name: arr / np.linalg.norm(arr) if normalize else arr
            for name, arr in zip(array_names, arrays)
        })

        # Start plot
        fig, ax = plt.subplots(figsize=(max(6, len(arrays) * 1.2), 5))
        # plt.subplots_adjust(top=0.85)

        sns.violinplot(
            data=df,
            inner="box",
            linewidth=1.2,
            palette=palette,
            alpha=alpha,
            ax=ax
        )

        # Add standard deviation and mean text
        for i, column in enumerate(df.columns):
            mean = np.mean(df[column]) if means is None else means[i]
            std = np.std(df[column], ddof=1) if stds is None else stds[i]
            ax.text(
                i,
                df[column].max() * 1.05,
                f"mean = {mean:.4f}\nstd = {std:.4f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        # Add pairwise F-test stars and connecting hlines
        star_height_factor = 1.15
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                stars = ""
                for threshold, star in zip([0.001, 0.01, 0.05], ["***", "**", "*"]):
                    result = self._variance_f_test(arrays[i], arrays[j], alpha=threshold)
                    if result["reject_null"]:
                        stars = star
                        break
                if stars:
                    # y-position for the annotation
                    y = max(df.iloc[:, [i, j]].max()) * star_height_factor
                    # draw hline between violins
                    ax.plot([i, j], [y, y], color="black", linewidth=1.0)
                    # add stars text above the line
                    x = (i + j) / 2
                    ax.text(
                        x,
                        y + 0.01 * y,  # tiny bump above the line
                        stars,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="black"
                    )
                    star_height_factor += 0.05  # bump height for next line/stars to avoid overlap

        if title: ax.set_title(title, fontsize=12)
        if xlabel: ax.set_xlabel(xlabel, fontsize=10)
        if ylabel: ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='x', rotation=xrotation)
        ax.tick_params(axis='y', rotation=yrotation)
        if log_scale:
            ax.set_yscale('log')

        sns.despine()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, format=output_path.suffix[1:], bbox_inches="tight")
        plt.close()

    def _variance_f_test(self, sample1: np.ndarray, sample2: np.ndarray, alpha: float = 0.05):
        """
        Perform a two-sided F-test to compare the variances of two samples.

        Parameters
        ----------
        sample1 : np.ndarray
            First sample (1D array-like), assumed to be normally distributed.
        sample2 : np.ndarray
            Second sample (1D array-like), assumed to be normally distributed.
        alpha : float, optional
            Significance level of the test (default is 0.05).

        Returns
        -------
        dict
            A dictionary containing:
            - 'f_statistic': The F statistic (ratio of variances).
            - 'p_value': Two-tailed p-value.
            - 'reject_null': Boolean indicating whether to reject the null hypothesis.
            - 'null_hypothesis': Text explaining the null hypothesis.
        """

        # Sample variances
        var1 = np.var(sample1, ddof=1)
        var2 = np.var(sample2, ddof=1)

        # Degrees of freedom
        df1 = len(sample1) - 1
        df2 = len(sample2) - 1

        # F-statistic: always put the larger variance in numerator for two-sided test
        if var1 >= var2:
            f_stat = var1 / var2
            dfn, dfd = df1, df2
        else:
            f_stat = var2 / var1
            dfn, dfd = df2, df1

        # Two-tailed p-value
        p_value = 2 * min(f.cdf(f_stat, dfn, dfd), 1 - f.cdf(f_stat, dfn, dfd))
        reject = p_value < alpha

        return {
            "f_statistic": f_stat,
            "p_value": p_value,
            "reject_null": reject,
            "null_hypothesis": "The two samples have equal variance"
        }

    def heatmap(self,
        data: pd.DataFrame,
        output_path: Path,
        row_names: Optional[List[str]] = None,
        col_names: Optional[List[str]] = None,
        normalize: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        palette: str = "default",
        log_scale: bool = False,
        xrotation: int = 0,
        yrotation: int = 0,
    ) -> None:
        """
        Generate and save a customizable heatmap from a pandas DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame where rows represent methods and columns represent metrics.
            Must contain a "Method" column to be used as index.
        output_path : Path
            The path (including filename) where the heatmap image will be saved.
        row_names : List[str], optional
            Custom row labels to display instead of the DataFrame index.
        col_names : List[str], optional
            Custom column labels to display instead of the DataFrame columns.
        normalize : bool, default=False
            Whether to normalize each column to the [0, 1] range for visual comparability.
        title : str, optional
            Title of the heatmap plot.
        xlabel : str, optional
            Label for the x-axis (columns/metrics).
        ylabel : str, optional
            Label for the y-axis (rows/methods).
        palette : str, default="default"
            Name of the color palette to use. If "default", uses a blue-red diverging palette.
        log_scale : bool, default=False
            Whether to log-transform the data before plotting (not yet implemented).
        xrotation : int, default=0
            Rotation angle for x-axis tick labels.
        yrotation : int, default=0
            Rotation angle for y-axis tick labels.

        Returns
        -------
        None
            Saves the heatmap to the specified output_path.
        """
        df = data.copy()

        # Set method names as index
        df.set_index("Method", inplace=True)

        # Optional normalization per column
        df_normalized = df.copy()
        if normalize:
            for col in df.columns:
                df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # Apply log scale (future expansion placeholder)
        if log_scale:
            raise NotImplementedError("Log scale transformation is not yet implemented.")

        # Override row and column labels if provided
        if row_names:
            df_normalized.index = row_names
        if col_names:
            df_normalized.columns = col_names

        # Choose a colormap
        if palette == "default":
            cmap = sns.diverging_palette(250, 30, l=65, center="light", as_cmap=True)
        else:
            cmap = palette

        # Create a figure
        plt.figure(figsize=(12, 14))
        sns.heatmap(df_normalized, annot=False, cmap=cmap, linewidths=0.5, linecolor="gray")

        # Set labels and title
        plt.title(title or "", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel(xlabel or "Metrics", fontsize=14)
        plt.ylabel(ylabel or "Methods", fontsize=14)

        # Ticks and layout
        plt.xticks(rotation=xrotation, ha="right")
        plt.yticks(rotation=yrotation)

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, format=output_path.suffix[1:], bbox_inches="tight")
        plt.close()

    def heatmap(
        data: pd.DataFrame,
        direction_vector: List[bool],
        output_path: Path,
        row_names: Optional[List[str]] = None,
        col_names: Optional[List[str]] = None,
        normalize: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        palette: str = "default",
        log_scale: bool = False,
        xrotation: int = 0,
        yrotation: int = 0,
        heatmap_row: bool = True,
        heatmap_col: bool = True,
        display_values: bool = False,
        sort_col_by_average: bool = True,
    ) -> None:
        data = data.copy()
        data.set_index("Method", inplace=True)

        if log_scale:
            data = np.log(data.replace(0, np.nan)).fillna(0)

        data_normalized = data.copy()
        if normalize:
            for col in data.columns:
                data_normalized[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

        for i, ascending in enumerate(direction_vector):
            if not ascending:
                data_normalized.iloc[i] = 1 - data_normalized.iloc[i]

        if sort_col_by_average:
            avg_row = data_normalized.mean(axis=0)
            sorted_cols = avg_row.sort_values(ascending=False).index.tolist()
            data_normalized = data_normalized[sorted_cols]

            if col_names:
                col_rename_map = dict(zip(data.columns, col_names))
                sorted_names = [col_rename_map[c] for c in sorted_cols if c in col_rename_map]
                data_normalized.columns = sorted_names

        if row_names:
            data_normalized.index = row_names

        if palette == "default":
            cmap = sns.diverging_palette(250, 30, l=65, center="light", as_cmap=True)
        else:
            cmap = palette

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if heatmap_row or heatmap_col:
            g = sns.clustermap(
                data_normalized,
                cmap=cmap,
                row_cluster=heatmap_row,
                col_cluster=heatmap_col,
                linewidths=0.5,
                linecolor="gray",
                figsize=(12, 14),
                xticklabels=True,
                yticklabels=True,
                annot=display_values,
            )

            g.ax_heatmap.set_title(title or "", fontsize=16, fontweight='bold', pad=15)
            g.ax_heatmap.set_xlabel(xlabel or "Metrics", fontsize=14)
            g.ax_heatmap.set_ylabel(ylabel or "Methods", fontsize=14)

            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=xrotation, ha="right")
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=yrotation)

            g.savefig(output_path, dpi=300, format=output_path.suffix[1:], bbox_inches="tight")
            plt.close()
        else:
            fig = plt.figure(figsize=(12, 14))
            sns.heatmap(
                data_normalized,
                annot=display_values,
                cmap=cmap,
                linewidths=0.5,
                linecolor="gray"
            )

            plt.title(title or "", fontsize=16, fontweight='bold', pad=15)
            plt.xlabel(xlabel or "Metrics", fontsize=14)
            plt.ylabel(ylabel or "Methods", fontsize=14)
            plt.xticks(rotation=xrotation, ha="right")
            plt.yticks(rotation=yrotation)
            plt.tight_layout()
            fig.savefig(output_path, dpi=300, format=output_path.suffix[1:], bbox_inches="tight")
            plt.close()

    """
    def heatmap(
        data: pd.DataFrame,
        direction_vector: List[bool],
        output_path: Path,
        row_names: Optional[List[str]] = None,
        col_names: Optional[List[str]] = None,
        normalize: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        palette: str = "default",
        log_scale: bool = False,
        xrotation: int = 0,
        yrotation: int = 0,
        heatmap_row: bool = True,
        heatmap_col: bool = True,
        display_values: bool = False,
        sort_col_by_average: bool = True,
    ) -> None:
        data = data.copy()
        data.set_index("Method", inplace=True)

        if log_scale:
            data = np.log(data.replace(0, np.nan)).fillna(0)

        data_normalized = data.copy()
        if normalize:
            for col in data.columns:
                data_normalized[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

        if row_names:
            data_normalized.index = row_names
        if col_names:
            data_normalized.columns = col_names

        if palette == "default":
            cmap = sns.diverging_palette(250, 30, l=65, center="light", as_cmap=True)
        else:
            cmap = palette

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if heatmap_row or heatmap_col:
            g = sns.clustermap(
                data_normalized,
                cmap=cmap,
                row_cluster=heatmap_row,
                col_cluster=heatmap_col,
                linewidths=0.5,
                linecolor="gray",
                figsize=(12, 14),
                xticklabels=True,
                yticklabels=True,
            )

            g.ax_heatmap.set_title(title or "", fontsize=16, fontweight='bold', pad=15)
            g.ax_heatmap.set_xlabel(xlabel or "Metrics", fontsize=14)
            g.ax_heatmap.set_ylabel(ylabel or "Methods", fontsize=14)

            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=xrotation, ha="right")
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=yrotation)

            g.savefig(output_path, dpi=300, format=output_path.suffix[1:], bbox_inches="tight")
            plt.close()
        else:
            fig = plt.figure(figsize=(12, 14))
            sns.heatmap(data_normalized, annot=False, cmap=cmap, linewidths=0.5, linecolor="gray")

            plt.title(title or "", fontsize=16, fontweight='bold', pad=15)
            plt.xlabel(xlabel or "Metrics", fontsize=14)
            plt.ylabel(ylabel or "Methods", fontsize=14)
            plt.xticks(rotation=xrotation, ha="right")
            plt.yticks(rotation=yrotation)
            plt.tight_layout()
            fig.savefig(output_path, dpi=300, format=output_path.suffix[1:], bbox_inches="tight")
            plt.close()
    """

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

    def bar_plot(self,
                 data, 
                 x_labels=None, 
                 plot_title=None, 
                 x_title=None, 
                 y_title=None, 
                 inverted=False, 
                 output_path=None):
        """
        Bar plot with annotated textboxes above each bar.

        Parameters
        ----------
        data : list of dict
            Each dict must contain at least 3 key-value pairs. The *first* value is used as the y-axis bar height.
            The entire dictionary is shown as an annotation above the bar, one 'key: value' per line.
        x_labels : list of str, optional
            X-axis labels. If None, bars are indexed numerically.
        x_title : str
            Title for the x-axis.
        y_title : str
            Title for the y-axis.
        inverted : bool
            Whether to invert the y-axis.
        """
        # Get bar heights (first value of each dict)
        heights = [list(d.values())[0] for d in data]

        # Get annotation text (multiline key: value per bar)
        annotations = [
            "\n".join([f"{k}: {v}" for k, v in d.items()])
            for d in data
        ]

        # Colorblind-friendly vibrant palette (ColorBrewer Set2-like)
        colors = [
            "#66c2a5", "#fc8d62", "#8da0cb",
            "#e78ac3", "#a6d854", "#ffd92f",
            "#e5c494", "#b3b3b3"
        ]

        # X-axis positions
        x_pos = list(range(len(data)))
        x_labels = x_labels or [str(i) for i in x_pos]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x_pos, heights, color=colors[:len(data)])

        # Annotate each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                annotations[i],
                ha='center',
                va='bottom',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", lw=0.5)
            )

        if plot_title is not None:
            ax.set_title(plot_title)
        if x_title is not None:
            ax.set_xlabel(x_title)
        if y_title is not None:
            ax.set_ylabel(y_title)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        if inverted:
            ax.invert_yaxis()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, format=output_path.suffix[1:], bbox_inches="tight")
        plt.close()

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

