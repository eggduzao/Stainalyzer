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

"""

###############################################################################
# Imports
###############################################################################

import os
import re
import yaml
import numpy as np
import pandas as pd
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

class TableReader:
    """
    TableReader class for loading various tabular file formats into a Pandas DataFrame.

    Supports: .csv, .tsv, .xlsx, .json, .parquet, .feather, .h5

    Attributes
    ----------
    path_name : str
        Path to the input file.
    first_header : bool
        If True, the first row will be treated as column headers.
    row_labels : bool
        If True, the first column will be treated as row labels.

    Methods
    -------
    get_pandas_dataframe()
        Loads the file into a Pandas DataFrame.
    _load_csv()
        Loads CSV/TSV files.
    _load_excel()
        Loads Excel files.
    _load_json()
        Loads JSON files.
    _load_parquet()
        Loads Parquet files.
    _load_feather()
        Loads Feather files.
    _load_hdf5()
        Loads HDF5 files.
    """

    def __init__(self, path_name=None, first_header=True, row_labels=False):
        """
        Initialize TableReader with file path and configuration options.

        Parameters
        ----------
        path_name : str, optional
            Path to the input file.
        first_header : bool, optional
            Whether the first row should be used as column headers (default is True).
        row_labels : bool, optional
            Whether the first column should be used as row labels (default is False).

        Notes
        -----
        The class automatically determines the file type based on the extension.
        If an unsupported extension is detected, an exception will be raised.
        """
        self.path_name = path_name
        self.first_header = first_header
        self.row_labels = row_labels

        # Static missing values
        self.missing_values_static = [
            None, np.nan, pd.NA,
            "null", "NULL", "None", "none",
            "na", "NA", "n/a", "N/A",
            "missing", "Missing",
            "unknown", "Unknown"
            #"not available", "Not Available",
            #"indisponível", "desconocido", "incognito",
            #"\\N", "\\n",
            #"NaN", "nan", "NAN",
            #"Inf", "-Inf", "infinity", "-infinity"
        ]

        # Regular expressions to catch repeating or symbolic patterns
        self.missing_patterns_re = [
            r"^\s+$",               # One or more whitespace characters
            #r"^-+$",                # One or more dashes
            #r"^\.+$",               # One or more dots
            #r"^,+$",                # One or more commas
            #r"^_+$",                # One or more underscores
            #r"^\?+$",               # One or more question marks
            r"^null$",              # Text-based null variants (case-insensitive)
            r"^(n/?a|na)$",         # NA, n/a variations
            r"^(unknown|missing)$", # Common textual representations
        ]
        self.missing_patterns_re = f"({'|'.join(self.missing_patterns_re)})"

        self.true_values = [
            "true", "True", "TRUE", "yes", "Yes", "YES",
            "verdadeiro", "Verdadeiro", "VERDADEIRO", "sim", "Sim", "SIM"
        ]

        self.false_values = [
            "false", "False", "FALSE", "no", "No", "NO",
            "falso", "Falso", "FALSO", "não", "Não", "NÃO", "nao", "Nao", "NAO"
        ]

    def get_pandas_dataframe(self):
        """
        Load the input file into a Pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame containing the file's content.

        Raises
        ------
        FileNotFoundError
            If the provided file path does not exist.
        ValueError
            If the file extension is unsupported.

        Examples
        --------
        >>> reader = TableReader('data.csv')
        >>> df = reader.get_pandas_dataframe()
        >>> print(df.head())
        """
        if not self.path_name or not os.path.exists(self.path_name):
            raise FileNotFoundError(f"File not found: {self.path_name}")

        file_extension = os.path.splitext(self.path_name)[1].lower()

        if file_extension in ['.csv', '.tsv']:
            df = self._load_csv()
        elif file_extension == '.xlsx':
            df = self._load_excel()
        elif file_extension == '.json':
            df = self._load_json()
        elif file_extension == '.parquet':
            df = self._load_parquet()
        elif file_extension == '.feather':
            df = self._load_feather()
        elif file_extension == '.h5':
            df = self._load_hdf5()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        # Apply regex-based missing values
        df = self._apply_regex_missing(df)
        return df

    def _load_csv(self):
        """
        Load CSV or TSV file into a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with CSV/TSV contents.
        """
        sep = '\t' if self.path_name.endswith('.tsv') else ','
        return pd.read_csv(self.path_name,
                           header=0 if self.first_header else None,
                           index_col=0 if self.row_labels else None,
                           sep=sep,
                           #dtype=str,  # Read everything as text
                           true_values=self.true_values,
                           false_values=self.false_values,
                           na_values=self.missing_values_static,  # Keep correct NaN detection
                           keep_default_na=True  # Ensure Pandas recognizes empty cells as NaN                           
                           )

    def _load_excel(self):
        """
        Load Excel file into a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with Excel file contents.
        """
        return pd.read_excel(self.path_name,
                             header=0 if self.first_header else None,
                             index_col=0 if self.row_labels else None,
                             #dtype=str,  # Read everything as text
                             true_values=self.true_values,
                             false_values=self.false_values,
                             na_values=self.missing_values_static,  # Keep correct NaN detection
                             keep_default_na=True  # Ensure Pandas recognizes empty cells as NaN
                             )

    def _load_json(self):
        """
        Load JSON file into a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with JSON file contents.
        """
        return pd.read_json(self.path_name)

    def _load_parquet(self):
        """
        Load Parquet file into a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with Parquet file contents.
        """
        return pd.read_parquet(self.path_name)

    def _load_feather(self):
        """
        Load Feather file into a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with Feather file contents.
        """
        return pd.read_feather(self.path_name)

    def _load_hdf5(self):
        """
        Load HDF5 file into a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with HDF5 file contents.
        """
        return pd.read_hdf(self.path_name)

    def _apply_regex_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies regex-based missing value replacements.

        Returns
        -------
        pd.DataFrame
            DataFrame with HDF5 file contents.
        """
        return df.replace(to_replace=self.missing_patterns_re, value=pd.NA, regex=True)

    def __repr__(self):
        """
        String representation of the TableReader instance.

        Returns
        -------
        str
            Developer-friendly representation.

        Examples
        --------
        >>> reader = TableReader('data.csv')
        >>> print(repr(reader))
        """
        return (f"TableReader(path_name={self.path_name!r}, "
                f"first_header={self.first_header}, "
                f"row_labels={self.row_labels})")

    def __str__(self):
        """
        User-friendly string representation.

        Returns
        -------
        str
            Descriptive information about the TableReader instance.

        Examples
        --------
        >>> reader = TableReader('data.csv')
        >>> print(str(reader))
        """
        return (f"TableReader for file: {self.path_name} | "
                f"First Header: {self.first_header} | "
                f"Row Labels: {self.row_labels}")
