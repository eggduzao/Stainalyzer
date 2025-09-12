"""
====================================================================================================
Stainalyzer Data Init Module
====================================================================================================

Overview
--------

Welcome to the OFFICIAL *Stainalyzer* API,
your one-stop **bioimage-analysis** dream tool,
powered by the ever-iconic GhostNet.

This init script serves GLAMOUR, serves GRACE,
and exposes the package's MAIN HIGH-LEVEL INTERFACES
so that you can import what you need like a queen.

-----
Exposed functions:
- segmentation: Tools for nuclear and cellular segmentation
- quantification: Stain quantification tools
- enhancement: Image enhancement like WOW
- classification: Machine-learning based phenotypic classifiers
-----

Ideal for:
- Biologists who slay
- Data scientists who want DRAMA in their plots
- Image nerds with a thirst for aesthetic segmentation

Examples
--------
>>> from stainalyzer import segment_nuclei
>>> segment_nuclei("cute_microscopy_image.tif")


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

 - Python Compatibility: Python 3.10 and above.

 - Required Packages: NumPy, SciPy and Open-Cv2.
 
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
------------
1. Placeholder.
2. Placeholder.

Typical Usage
-------------
$ python -m stainalyzer --help
$ python -m stainalyzer --input my_folder/ --task segment

Future Goals
------------
- Support CLI arguments
- Load GUI instead of CLI (switchable)
- Slay all TIFFs in one run

=======================================
Author: Eduardo Gade Gusmao           |
Created On: 11/12/2024                |
Last Updated: 15/02/2025              |
Version: 0.1.3                        |
License: <Currently_Withheld>         |
=======================================
=======================================
=======================================
"""

# Data packages
from stainalyzer.__version__ import __version__
from stainalyzer.data.pixel import Pixel
from stainalyzer.data.loader import MetricTables
from stainalyzer.data.loader_tester import LoaderTester
from stainalyzer.data.input_data import TableReader
from stainalyzer.data.loader_visualization import VizPlots

# All packages
__all__ = [
		   "__version__",
		   "Pixel",
		   "MetricTables",
		   "LoaderTester",
		   "TableReader",
		   "VizPlots",
		   ]
