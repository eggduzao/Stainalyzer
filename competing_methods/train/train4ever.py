"""
====================================================================================================
io.py - General Module for I/O Processing, Analysis, and Support
====================================================================================================

Overview
--------
This module, 'io.py', placeholder.


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

    # from matplotlib.font_manager import findSystemFonts

    # for font in findSystemFonts():
    #     font_path = Path(font)
    #     if "Consolas" == font_path.stem:
    #         print(f"Consolas Stem = {font}")

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
import matplotlib.pyplot as plt
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

    def violin_plot(self,
                    *arrays: np.ndarray,
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
        """
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
        """
        # Prepare the data
        df = pd.DataFrame({
            f"Array {i+1}": arr / np.linalg.norm(arr) if normalize else arr
            for i, arr in enumerate(arrays)
        })

        # Start plot
        fig, ax = plt.subplots(figsize=(max(6, len(arrays) * 1.2), 5))

        # Set font to Consolas
        plt.rcParams["font.family"] = "Consolas"

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
                f"mean = {mean:.2f}\nstd = {std:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontfamily="Consolas"
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

