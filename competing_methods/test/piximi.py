"""
====================================================================================================
utils.py - General Utility Module for Data Processing, Analysis, and Software Infrastructure Support
====================================================================================================

Overview
--------
This module, `utils.py`, serves as a centralized collection of utility functions and classes designed 
to support various aspects of software development, data processing, and analysis tasks. The utilities 
provided herein are intended to offer reusable, maintainable, and efficient solutions to common 
challenges encountered during application development, including but not limited to file I/O operations, 
string manipulation, data validation, configuration handling, performance monitoring, and logging. 

The primary goal of this module is to abstract away repetitive, low-level operations, enabling developers 
to focus on domain-specific logic while ensuring code consistency, clarity, and reliability. The functions 
and classes have been implemented with readability, performance optimization, and compatibility with 
industry-standard Python practices, including adherence to PEP 8 style guidelines and utilization of 
extensive inline documentation following the NumPy docstring format.

Intended Use Cases
-------------------
- **Configuration Management:** Reading, writing, and validating configuration files in formats such as YAML, 
  JSON, and INI for flexible application settings and parameters.
- **Data Manipulation and Analysis:** Providing utility methods for efficient data type conversions, filtering, 
  sorting, and aggregation operations.
- **Logging and Debugging:** Facilitating structured logging, exception tracking, and performance profiling to 
  enhance software maintainability and debugging efficiency.
- **File System Operations:** Simplifying common tasks such as directory creation, file existence checks, and 
  path manipulations with cross-platform compatibility.
- **Validation and Rule Processing:** Supporting rule-based validation frameworks with dynamic conditions, 
  including type assertions, range checks, and custom logical rules.

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

**Example 1: Configuration Management**

```python
from utils import ConfigLoader**

config = ConfigLoader('config.yaml').load()
print(f"Database Host: {config['database']['host']}")

**Example 2: Data Validation with Rule Engine**
```

```python
from utils import RuleEngine

rule = RuleEngine('age', condition='>=', threshold=18)
result = rule.validate(25)  # Returns True
```

**Example 3: File System Operations**

```python
from utils import FileHandler

file_path = 'data/input.csv'
if FileHandler.exists(file_path):
    content = FileHandler.read(file_path)
    print(content)
```

Development Notes:
-------------

 - Python Compatibility: Python 3.10 and above.

 - Required Packages: PyYAML, NumPy, Pandas, logging.
 
 - Testing Framework: The module includes unit tests implemented using pytest to ensure reliability
across different Python environments.
 
 - Code Style Compliance: All code follows PEP 8 guidelines, with additional comments to aid
maintainability for future developers.

Manual:
-------

1. Ensure environment variables are set before YAML loading.
2. Structure YAML rules like:
   - Boolean rule: !rule {name: "is_senior", condition: "age > 60", rule_type: "boolean", parameters: {expected: true}}
   - Numeric rule: !rule {name: "age_range", condition: "18 <= age <= 99", rule_type: "numeric", parameters: {min: 18, max: 99}}
   - String rule: !rule {name: "username_length", condition: "3 <= len(username) <= 15", rule_type: "string", parameters: {min_len: 3, max_len: 15}}
   - Category rule: !rule {name: "valid_department", condition: "department in [Sales, HR]", rule_type: "category", parameters: {allowed: ["Sales", "HR"]}}

Usage Example:
--------------
os.environ['SERVER_HOST'] = "localhost"
os.environ['SERVER_PORT'] = "8080"

with open('config.yaml') as file:
    config = yaml.load(file, Loader=UnifiedLoader)
print(config)

Rule Evaluation Example:
------------------------
rule = Rule(name="test", condition="value > 5", rule_type="numeric", parameters={"min": 5})
print(rule.evaluate(6))  # True
print(rule.evaluate(3))  # False

Integration with YamlUtils:
----------------------------
1. Replace yaml.safe_load with yaml.load using UnifiedLoader.
2. Call rules by parsing YAML and creating Rule objects dynamically.
3. Extend YamlUtils with rule validation methods.

EXAMPLE YAML STRUCTURE:
-----------------------
server_config:
  host: "${SERVER_HOST}"
  port: "${SERVER_PORT}"

rules:
  - !rule {name: "age_check", condition: ">=18", rule_type: "numeric", parameters: {min: 18}}
  - !rule {name: "username_length", condition: "len >= 5", rule_type: "string", parameters: {min_len: 5}}
  - !rule {name: "senior_check", condition: "age > 60", rule_type: "boolean", parameters: {expected: true}}
  - !rule {name: "department_valid", condition: "in valid departments", rule_type: "category", parameters: {allowed: ["HR", "Sales"]}}

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
from typing import List, Any, Dict

from .exceptions import YAMLValidationError

###############################################################################
# Constants
###############################################################################

# Constants
SEED = 1987
np.random.seed(SEED)

###############################################################################
# Classes
###############################################################################

class PandasUtils:

    """
    PandasUtils class for further Pandas DataFrame methods.

    Supports: Pandas DataFrame Class

    Methods
    -------
    slice_dataframe()
        Slices any Pandas Datagrame into a Pandas DataFrame.
    """

    def __init__(self):
        """
        Initialize PandasUtils

        Parameters
        ----------

        Notes
        -----
        The class automatically determines the file type based on the extension.
        If an unsupported extension is detected, an exception will be raised.
        """

    def slice_dataframe(self,
                        df: pd.DataFrame,
                        row_indices: List[Any],
                        col_indices: List[Any],
                        row_label_int: bool = False,
                        col_label_int: bool = False
                        ) -> pd.DataFrame | None:
        """
        Slices a pandas DataFrame using specified rows and columns.

        This function accepts lists of indices or labels to extract specific rows and columns. It
        ensures that the result is always a DataFrame, even for a single cell, and returns None if empty.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to slice.
        row_indices : List[Any]
            A list containing the row indices or labels to select.
        col_indices : List[Any]
            A list containing the column indices or labels to select.
        row_label_int : bool, optional
            If True, interprets integers in `row_indices` as positional indices (default: False).
        col_label_int : bool, optional
            If True, interprets integers in `col_indices` as positional indices (default: False).

        Returns
        -------
        pd.DataFrame or None
            A DataFrame containing the selected rows and columns, or None if the result is empty.

        Raises
        ------
        ValueError
            If invalid row or column indices are provided.

        Example
        -------
        data = {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "disease": ["Flu", "Cold", "Cancer", "Diabetes", "Asthma"],
            "cardiac_status": ["True", None, "True", "False", None]
        }

        # Create sample DataFrame with integer index
        df = pd.DataFrame(data)

        # Valid examples
        print(slice_dataframe(df, [1, 3], ["name", "disease"]))
        print(slice_dataframe(df, [0, 2, 4], ["name", "cardiac_status"]))

        # Using integers as indices
        print(slice_dataframe(df, [0, 2, 4], [0, 2],
                              row_label_int=True, col_label_int=True))

        # Single cell extraction
        print(slice_dataframe(df, [1], ["name"]))

        # Invalid cases (will raise ValueError)
        try:
            print(slice_dataframe(df, [100], ["name"]))  # Invalid row index
        except ValueError as e:
            print(e)

        try:
            print(slice_dataframe(df, [1], ["nonexistent_column"]))  # Invalid column name
        except ValueError as e:
            print(e)
        """

        # Validate row indices
        try:
            if row_label_int:
                row_selection = df.iloc[row_indices]
            else:
                row_selection = df.loc[row_indices]
        except Exception as e:
            raise ValueError(f"Invalid row indices provided: {row_indices}. Error: {e}")

        # Validate column indices
        try:
            if col_label_int:
                result = row_selection.iloc[:, col_indices]
            else:
                result = row_selection.loc[:, col_indices]
        except Exception as e:
            raise ValueError(f"Invalid column indices provided: {col_indices}. Error: {e}")

        # Ensure result is always a DataFrame, even if single cell
        if result.shape == (1, 1):
            single_cell_df = pd.DataFrame(result.values, index=[row_indices[0]], columns=[col_indices[0]])
            return single_cell_df

        return result if not result.empty else None

    def slice_rows(self, df: pd.DataFrame, start: int, end: int) -> pd.DataFrame | None:
        sliced = df.iloc[start:end]
        return sliced if not sliced.empty else None

    def slice_columns(self, df: pd.DataFrame, start: int, end: int) -> pd.DataFrame | None:
        sliced = df.iloc[:, start:end]
        return sliced if not sliced.empty else None

    def slice_rows_by_label(self, df: pd.DataFrame, start_label: Any, end_label: Any) -> pd.DataFrame | None:
        sliced = df.loc[start_label:end_label]
        return sliced if not sliced.empty else None

    def slice_columns_by_label(self, df: pd.DataFrame, start_label: Any, end_label: Any) -> pd.DataFrame | None:
        sliced = df.loc[:, start_label:end_label]
        return sliced if not sliced.empty else None

    def slice_with_dictionary(self, data: pd.DataFrame, dictionary: Dict) -> pd.DataFrame:
        """
        Slice a Pandas DataFrame to contain only the columns specified as keys in a given dictionary.

        This function extracts a subset of the DataFrame by selecting only the columns that exist as keys 
        in the provided dictionary. If a key in the dictionary does not match any column in `data`, 
        it will be ignored.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame from which columns should be selected.

        dictionary : Dict
            A dictionary where **keys represent the column names to keep** in the returned DataFrame.
            The values of the dictionary are **ignored**, meaning only the dictionary keys matter.

        Returns
        -------
        pd.DataFrame
            A new DataFrame containing only the selected columns.

        Notes
        -----
        - This method **does not modify the original DataFrame**; it returns a new sliced DataFrame.
        - Columns that **do not exist in the original DataFrame but are present in the dictionary keys** 
          will be ignored.
        - This function is **case-sensitive** (i.e., "ColumnA" and "columna" are treated as different).

        Example
        -------
        >>> import pandas as pd
        >>> from typing import Dict
        >>> data = pd.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6],
        ...     "C": [7, 8, 9]
        ... })
        >>> dictionary = {"A": "some_value", "C": "another_value", "D": "not_in_data"}
        >>> utils = PandasUtils()
        >>> sliced_df = utils.slice_with_dictionary(data, dictionary)
        >>> print(sliced_df)
           A  C
        0  1  7
        1  2  8
        2  3  9
        """

        # Step 1: Extract the column names from the DataFrame
        existing_columns = set(data.columns)  # Get all existing columns in the DataFrame
        
        # Step 2: Extract the dictionary keys (i.e., the desired columns to keep)
        selected_columns = set(dictionary.keys())  # Extract dictionary keys (column names)
        
        # Step 3: Identify the intersection (columns that exist in both `data` and `dictionary.keys()`)
        valid_columns = existing_columns.intersection(selected_columns)  # Only keep existing columns

        # Step 4: Slice the DataFrame using only the valid columns and return the result
        sliced_data = data[list(valid_columns)]

        #print(f"Sliced DataFrame Shape: {sliced_data.shape}")  # Verbose output of final shape

        return sliced_data

class ExcelUtils:
    """
    A utility class for handling Excel file operations, including header extraction and tabular beautification.

    Methods
    -------
    __repr__()
        Returns a developer-friendly string representation of the object.
    __str__()
        Returns a user-friendly string representation of the object.
    get_headers(input_path, output_path=None)
        Extracts and prints or saves the headers of an Excel file.
    tab_beautify(input_path, output_path=None, columns=None, max_num_characters=20)
        Beautifies and prints or saves the tabular content of an Excel file with column alignment.
    """

    def __init__(self):
        """
        Initialize the ExcelUtils class.

        Notes
        -----
        This class supports operations like extracting headers and beautifying tables from Excel files.
        """
        pass

    def __repr__(self):
        """
        Returns a developer-friendly string representation of the object.

        Returns
        -------
        str
            Developer-friendly representation of the object.
        """
        return "ExcelUtils()"

    def __str__(self):
        """
        Returns a user-friendly string representation of the object.

        Returns
        -------
        str
            User-friendly representation of the object.
        """
        return "Excel Utilities class for Excel file operations"

    def clean_string(self, text: str) -> str:
        """
        Cleans a string by removing excessive spaces, newlines, and unnecessary quotes.
        
        Parameters
        ----------
        text : str
            The input string to be cleaned.

        Returns
        -------
        str
            A single-line, cleaned-up string.
        """
        if not isinstance(text, str):
            return str(text)  # Ensure conversion for non-string values

        text = text.strip()  # Remove leading and trailing spaces
        text = text.replace("\n", " ").replace("\\n", " ")  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = text.strip("'\"")  # Remove unnecessary surrounding quotes
        
        return text

    def excel_to_yaml(self,
                      input_path: str,
                      yaml_key: int,
                      yaml_values: list[int],
                      yaml_level: int = 2,
                      output_path: str = None):
        """
        Converts an Excel spreadsheet into a YAML-formatted output.

        Parameters
        ----------
        input_path : str
            Path to the Excel file to be read.
        yaml_key : int
            1-based index of the column that will serve as the YAML key.
        yaml_values : list[int]
            List of 1-based indices of the columns to be included as values in YAML.
        yaml_level : int, optional
            Indentation level for structuring the YAML output (default is 2).
        output_path : str, optional
            If specified, writes the YAML output to this file instead of printing.

        Raises
        ------
        FileNotFoundError
            If the input Excel file does not exist.
        ValueError
            If the specified column indices are out of range.

        Returns
        -------
        None
            Prints the YAML structure or writes to a file.
        """
        
        # Load the spreadsheet
        try:
            df = pd.read_excel(input_path, dtype=str)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {input_path}")

        # Convert 1-based indices to 0-based indices for pandas
        yaml_key -= 1
        yaml_values = [col - 1 for col in yaml_values]

        # Validate column indices
        if yaml_key >= len(df.columns) or any(col >= len(df.columns) for col in yaml_values):
            raise ValueError("Provided column indices exceed the number of columns in the spreadsheet.")

        # YAML structure initialization
        yaml_output = {}
        table_counter = 1
        table_key = f"table{table_counter}"
        yaml_output[table_key] = {}

        # Regular expression for valid YAML keys
        yaml_key_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")

        # Process each row
        for _, row in df.iterrows():

            # Identify merged cells (if only one non-null value exists in a row)
            non_null_values = row.dropna()
            if len(non_null_values) == 1 and row.isna().sum() == (len(row) - 1):
                merged_value = self.clean_string(non_null_values.iloc[0])  # Get the single merged value
                key = self.clean_string(row.iloc[0])  # The first column's value as key
                yaml_output[table_key][key] = merged_value
                continue

            # Extract key and ensure it's properly formatted
            key = self.clean_string(row.iloc[yaml_key])
            if not yaml_key_pattern.match(key):
                print(f"Warning: Key '{key}' does not follow YAML naming conventions.")

            # Extract and clean values
            values = [self.clean_string(row.iloc[col]) for col in yaml_values]
            
            # Ensure all values are enclosed in double quotes
            formatted_values = [f'{v}' for v in values]

            # Add to YAML dictionary
            yaml_output[table_key][key] = formatted_values

        # Output YAML content
        if output_path:
            output_file = open(output_path, "w", encoding="utf-8")
            prefix = ""
            for i in range(1, yaml_level):
                output_file.write(f"{prefix}table{i}:\n")
                prefix += "  "
            for key, item in yaml_output[table_key].items():
                if key=="nan" or key==None:
                    continue
                else:
                    output_file.write(f"{prefix}{key}: {item}\n")
        else:
            prefix = ""
            for i in range(1, yaml_level):
                print(f"{prefix}table{i}:\n")
                prefix += "  "
            for key, item in yaml_output[table_key].items():
                if key=="nan" or key==None:
                    continue
                else:
                    print(f"{prefix}{key}: {item}\n")

    def get_headers(self, input_path, output_path=None):
        """
        Extracts and prints or saves the headers of an Excel file.

        Parameters
        ----------
        input_path : str
            Path to the Excel file.
        output_path : str, optional
            Path to save the headers if provided; otherwise, headers are printed.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        ValueError
            If the file is not a valid Excel file or cannot be read.

        Notes
        -----
        This method reads only the first row of the Excel file to retrieve the headers,
        then either prints them to the console or writes them to a specified output file.
        """
        try:
            # Step 1: Validate that the input file exists before proceeding
            if not os.path.exists(input_path):
                # If the file does not exist, raise an exception and provide a clear error message
                raise FileNotFoundError(f"File not found: {input_path}")

            # Step 2: Attempt to read the first row of the Excel file to get the headers
            try:
                # Use pandas to read only one row; column names represent the headers
                df = pd.read_excel(input_path, nrows=1)
            except Exception as e:
                # If pandas fails to read the file, raise a ValueError with the underlying error message
                raise ValueError(f"Failed to read the Excel file. Ensure it's a valid Excel format: {e}")

            # Step 3: Extract the headers from the DataFrame as a list of strings
            headers = df.columns.tolist()

            # Step 4: Determine output method based on the presence of an output path
            if output_path:
                # If an output path is specified, write the headers to the specified file
                try:
                    with open(output_path, 'w') as f:
                        # Iterate over each header and write it as a separate line in the file
                        for header in headers:
                            f.write(f"{header}\n")
                    # Inform the user that the headers have been successfully written to the file
                    print(f"Headers successfully written to {output_path}")
                except Exception as e:
                    # Handle file write errors and provide feedback to the user
                    raise ValueError(f"Failed to write headers to file: {e}")
            else:
                # If no output path is specified, print the headers to the console
                print("Headers:")
                for header in headers:
                    print(header)

        except FileNotFoundError as e:
            # Re-raise the FileNotFoundError with the original message for clarity
            raise e
        except ValueError as e:
            # Re-raise ValueError if there was an issue reading the Excel file or writing the headers
            raise e
        except Exception as e:
            # Catch any unexpected exceptions and raise them with a descriptive message
            raise RuntimeError(f"An unexpected error occurred while extracting headers: {e}")

    def _strip_cell(self, to_strip, wrap_length):
        """
        Splits a string into chunks of length wrap_length - 1, respecting word boundaries.

        If a word is larger than wrap_length, it will be split into syllable-based chunks with a trailing hyphen.

        Parameters
        ----------
        to_strip : str
            The string to be split into chunks.
        wrap_length : int
            The maximum allowed length per chunk (except when splitting long words).

        Returns
        -------
        list of str
            List of string chunks forming the original string when concatenated.

        Raises
        ------
        ValueError
            If wrap_length is less than 2.

        Notes
        -----
        - Portuguese syllable-based splitting prioritizes splitting after vowels.
        - Words are not broken unless necessary (if exceeding wrap_length).
        - Non-alphanumeric characters are considered valid splitting points.
        """

        if wrap_length < 2:
            raise ValueError("wrap_length must be at least 2")

        chunks = []
        words = to_strip.split()
        current_line = ""

        def syllable_split(word, limit):
            """Splits a single word into syllable-based chunks with a trailing hyphen if needed."""
            vowels = "aeiouAEIOU"
            result = []
            while len(word) > limit:
                # Find last vowel before limit
                split_pos = limit - 1
                for i in range(split_pos, 0, -1):
                    if word[i] in vowels:
                        split_pos = i + 1
                        break
                # Adjust for long words with no vowels
                split_pos = max(1, min(split_pos, limit - 1))
                result.append(word[:split_pos] + "-")
                word = word[split_pos:]
            result.append(word)
            return result

        for word in words:
            # If adding the word exceeds the line length
            if len(current_line) + len(word) + 1 > wrap_length:
                if current_line:
                    chunks.append(current_line.strip())
                    current_line = ""

            # Handle long words
            if len(word) >= wrap_length:
                split_word = syllable_split(word, wrap_length)
                for part in split_word:
                    if len(current_line) + len(part) + 1 > wrap_length:
                        if current_line:
                            chunks.append(current_line.strip())
                            current_line = ""
                    current_line += part + " "
                continue

            # Add normal words
            current_line += word + " "

        if current_line:
            chunks.append(current_line.strip())

        return chunks

    def tab_beautify(self, input_path, output_path=None, columns=None, max_num_characters=20, max_char_output_line=240, tab_length=4):
        """
        Beautifies and prints or saves the tabular content of an Excel file with aligned columns.

        Parameters
        ----------
        input_path : str
            Path to the Excel file.
        output_path : str, optional
            Path to save the formatted table; if None, prints to the console.
        columns : list of int, optional
            List of column indices (1-based) to include. If None, includes all columns.
        max_num_characters : int, optional
            Maximum number of characters per cell. Longer content is truncated.
        max_char_output_line : int, optional
            Maximum characters per line.
        tab_length : int, optional
            Number of spaces per tab.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        ValueError
            If parameters are inconsistent or the file cannot be read.

        Notes
        -----
        - max_num_characters should ideally be 1 less than a multiple of tab_length.
        - max_char_output_line should ideally be a multiple of tab_length.
        - tab_length should be less than max_num_characters and max_num_characters should be less than max_char_output_line/2.
        """

        # Validate parameters
        if tab_length >= max_num_characters or max_num_characters >= max_char_output_line // 2:
            raise ValueError("Invalid parameters: Ensure tab_length < max_num_characters < max_char_output_line/2")

        try:
            # Step 1: Load the Excel file
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"File not found: {input_path}")
            df = pd.read_excel(input_path, dtype=str).fillna("")

            # Step 2: Filter columns if specified
            if columns:
                df = df.iloc[:, [col - 1 for col in columns if 1 <= col <= len(df.columns)]]

            # Step 3: Calculate max widths for each column
            column_widths = []
            for col in df.columns:
                max_width = max(len(str(cell)) for cell in df[col])
                column_widths.append(min(max_width, max_num_characters))

            # Step 4: Prepare header
            headers = [str(header)[:max_num_characters] for header in df.columns]
            column_widths = [max(len(header), width) for header, width in zip(headers, column_widths)]

            # Step 5: Prepare wrapped text matrix
            wrapped_matrix = []
            for _, row in df.iterrows():
                wrapped_row = [self._strip_cell(str(cell), max_num_characters) for cell in row]
                wrapped_matrix.append(wrapped_row)

            # Step 6: Generate lines for tabular display
            lines = []

            # Header
            header_line = "\t".join(f"{header:<{width}}" for header, width in zip(headers, column_widths))
            lines.append(header_line)
            lines.append("-" * min(len(header_line), max_char_output_line))

            # Data rows
            max_lines = max(len(max(cell, key=len) if cell else "") for row in wrapped_matrix for cell in row)
            for i, row in enumerate(wrapped_matrix):
                max_line_count = max(len(cell) for cell in row)
                for line_idx in range(max_line_count):
                    formatted_line = []
                    for cell_idx, cell_lines in enumerate(row):
                        line = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                        formatted_line.append(f"{line:<{column_widths[cell_idx]}}")
                    lines.append("\t".join(formatted_line)[:max_char_output_line])

            # Step 7: Output the formatted table
            output = "\n".join(lines)
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(output)
            else:
                print(output)

        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Failed to process and beautify the Excel file: {e}")

    def pandas_beautify(self, df, output_path=None, columns=None, max_num_characters=20, max_char_output_line=240, tab_length=4):
        """
        Beautifies and prints or saves the tabular content of an Excel file with aligned columns.

        Parameters
        ----------
        df : pd.DataFrame
            A pandas dataframe
        output_path : str, optional
            Path to save the formatted table; if None, prints to the console.
        columns : list of int, optional
            List of column indices (1-based) to include. If None, includes all columns.
        max_num_characters : int, optional
            Maximum number of characters per cell. Longer content is truncated.
        max_char_output_line : int, optional
            Maximum characters per line.
        tab_length : int, optional
            Number of spaces per tab.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        ValueError
            If parameters are inconsistent or the file cannot be read.

        Notes
        -----
        - max_num_characters should ideally be 1 less than a multiple of tab_length.
        - max_char_output_line should ideally be a multiple of tab_length.
        - tab_length should be less than max_num_characters and max_num_characters should be less than max_char_output_line/2.
        """

        # Validate parameters
        if tab_length >= max_num_characters or max_num_characters >= max_char_output_line // 2:

            raise ValueError("Invalid parameters: Ensure tab_length < max_num_characters < max_char_output_line/2")

        try:

            # Step 1: Filter columns if specified
            if columns:
                df = df.iloc[:, [col - 1 for col in columns if 1 <= col <= len(df.columns)]]

            # Step 2: Calculate max widths for each column
            column_widths = []
            for col in df.columns:
                max_width = max(len(str(cell)) for cell in df[col])
                column_widths.append(min(max_width, max_num_characters))

            # Step 3: Prepare header
            headers = [str(header)[:max_num_characters] for header in df.columns]
            column_widths = [max(len(header), width) for header, width in zip(headers, column_widths)]

            # Step 4: Prepare wrapped text matrix
            wrapped_matrix = []
            for _, row in df.iterrows():
                wrapped_row = [self._strip_cell(str(cell), max_num_characters) for cell in row]
                wrapped_matrix.append(wrapped_row)

            # Step 5: Generate lines for tabular display
            lines = []

            # Header
            header_line = "\t".join(f"{header:<{width}}" for header, width in zip(headers, column_widths))
            lines.append(header_line)
            lines.append("-" * min(len(header_line), max_char_output_line))

            # Data rows
            max_lines = max(len(max(cell, key=len) if cell else "") for row in wrapped_matrix for cell in row)
            for i, row in enumerate(wrapped_matrix):
                max_line_count = max(len(cell) for cell in row)
                for line_idx in range(max_line_count):
                    formatted_line = []
                    for cell_idx, cell_lines in enumerate(row):
                        line = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                        formatted_line.append(f"{line:<{column_widths[cell_idx]}}")
                    lines.append("\t".join(formatted_line)[:max_char_output_line])

            # Step 7: Output the formatted table
            output = "\n".join(lines)
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(output)
            else:
                print(output)

        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Failed to process and beautify the Excel file: {e}")

