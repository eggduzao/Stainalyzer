"""
====================================================================================================
data.py - General Module for I/O Processing, Analysis, and Support
====================================================================================================

Overview
--------
This module, 'data.py', placeholder.


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

# Example Usage TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
op = DataOperations()
print(op.compare(5, np.int64(5)))  # True, since allow_numpy_numeric is True by default


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
from datetime import datetime, timedelta
from collections import OrderedDict, namedtuple, deque
from collections.abc import Iterable
from typing import List, Any, Dict, Tuple

###############################################################################
# Constants
###############################################################################

# Constants
SEED = 1987
np.random.seed(SEED)

###############################################################################
# Classes
###############################################################################


class DataOperations:
    """
    A utility class for performing various data operations, including comparison, 
    string representations, and other functions that might be added in the future.

    This class provides a structured way to handle complex data operations while 
    maintaining clarity, readability, and adherence to best coding practices.

    Attributes
    ----------
    None (for now)
    
    Methods
    -------
    __init__():
        Initializes an instance of the DataOperations class.
    
    compare(a, b) -> bool:
        Compares two values with strict type checking and custom comparison rules.
    
    __repr__() -> str:
        Returns a developer-friendly string representation of the object.
    
    __str__() -> str:
        Returns a user-friendly string representation of the object.
    """

    def __init__(self):
        """
        Initializes the DataOperations class.

        This method is currently empty but serves as a placeholder for future 
        attributes or initialization logic.
        """
        pass  # No attributes yet, but can be extended later

    def is_in(self, value: Any, iterable: Any, **kwargs) -> bool:
        """
        Placeholder.

        Parameters
        ----------
        value : Any
            The value to compare.
        iterable : Any
            The iterable to verify "is in".
        kwargs : bool
            All other flags to pass to self.compare().

        Returns
        -------
        bool
            bool: True if value is in iterable.

        Notes
        -----
        - Placeholder.
        """

        # Check if iterable is really an iterable
        if not self._is_iterable(iterable):
            raise TypeError(f"Expected an iterable, but got {type(iterable).__name__}")

        # Check if value is iterable
        is_iterable = False
        list_like = True
        if self._is_iterable(value):
            is_iterable = True
            if isinstance(value, (dict, OrderedDict)):
                list_like = False

        # Compare each value and return True when found
        if isinstance(iterable, (list, tuple, range, set, frozenset, deque, np.ndarray, pd.Series, pd.Index)):

            if not is_iterable:
                for obj in iterable:
                    if self.compare(value, obj, **kwargs):
                        return True
            else:
                for obj1 in iterable:
                    if list_like:
                        for obj2 in value:
                            if self.compare(obj1, obj2, **kwargs):
                                return True
                    else:
                        for obj2 in value.keys():
                            if self.compare(obj1, obj2, **kwargs):
                                return True

        # Dictionaries are a special case as "is in" compares keys, not values.
        if isinstance(iterable, (dict, OrderedDict)):

            if not is_iterable:
                for key in iterable.keys():
                    if self.compare(value, key, **kwargs):
                        return True
            else:
                for obj1 in iterable.keys():
                    if list_like:
                        for obj2 in value:
                            if self.compare(obj1, obj2, **kwargs):
                                return True
                    else:
                        for obj2 in value.keys():
                            if self.compare(obj1, obj2, **kwargs):
                                return True

        return False              

    def is_not_in(self, value: Any, iterable: Any, **kwargs) -> bool:
        """
        Placeholder.

        Parameters
        ----------
        value : Any
            The value to compare.
        iterable : Any
            The iterable to verify "is not in".
        kwargs : bool
            All other flags to pass to self.compare().

        Returns
        -------
        bool
            bool: True if value is not in iterable.

        Notes
        -----
        - Placeholder.
        """

        # Check if iterable is really an iterable

        return self.is_in(value, iterable, **kwargs)  

    def compare(self,
                value1: Any,
                value2: Any,
                allow_numpy_numeric: bool = True,
                allow_pandas_numeric: bool = True,
                allow_numeric_types: bool = True,
                allow_numpy_string: bool = True,
                allow_equal_inf: bool = True,
                allow_numpy_inf: bool = True,
                allow_numpy_datetime: bool = True,
                allow_pandas_datetime: bool = True,
                allow_numpy_sequence: bool = False,
                allow_pandas_sequence: bool = False,
                allow_numpy_others: bool = False,
                allow_pandas_others: bool = False) -> bool:
        """
        Compares two values under standardized conditions, considering native
        Python, NumPy, and Pandas types.

        This function will be implemented later to allow more complex logic
        on sequence/mapping types

        Parameters
        ----------
        value1 : Any
            The first value to compare.
        value2 : Any
            The second value to compare.
        allow_numpy_numeric : bool
            Allow NumPy numeric types to be equivalent to Python scalars.
        allow_pandas_numeric : bool
            Allow Pandas numeric types to be equivalent to Python scalars.
        allow_numeric_types : bool
            Allow comparisons between different numeric types (e.g., int vs. float).
        allow_numpy_string : bool
            Allow NumPy string types to be equivalent to Python str.
        allow_equal_inf : bool
            Allow Inf and -Inf to be treated as equal.
        allow_numpy_inf : bool
            Allow NumPy infinity values to be treated as equal to Python infinity.
        allow_numpy_datetime : bool
            Allow NumPy datetime64 to be equivalent to Python datetime.
        allow_pandas_datetime : bool
            Allow Pandas Timestamp to be equivalent to Python datetime.
        allow_numpy_sequence : bool
            Allow NumPy arrays to be equivalent to Python lists.
        allow_pandas_sequence : bool
            Allow Pandas sequences (Series, Index, Categorical) to be equivalent to Python sequences.
        allow_numpy_others : bool
            Allow NumPy-specific types to be compared (e.g., np.object_).
        allow_pandas_others : bool
            Allow Pandas-specific objects (e.g., DataFrame) to be compared.

        Returns
        -------
        bool
            bool: True if values are considered equal under the defined conditions, otherwise False.

        Notes
        -----
        - Future implementations might include:
            - Sequence/Mapping types
            - Handling of pandas/numpy objects.
            - Custom equivalence rules for different data types with YAML.
            - Configurable flags for flexible comparison.
        """

        # Handle None and missing value types
        if value1 in {None, np.nan, pd.NA} and value2 in {None, np.nan, pd.NA}:
            return True
        
        # Handle booleans
        if isinstance(value1, (bool, np.bool_)) and isinstance(value2, (bool, np.bool_)):
            return value1 == value2
        # If one of the values is a boolean and the other isn't -> They need to be different
        elif isinstance(value1, (bool, np.bool_)) or isinstance(value2, (bool, np.bool_)): 
            return False
        
        # Handle numeric types
        if allow_numpy_numeric or allow_pandas_numeric:
            if isinstance(value1, (int, float, complex, np.number, pd.Int64Dtype, pd.Float64Dtype)) and \
               isinstance(value2, (int, float, complex, np.number, pd.Int64Dtype, pd.Float64Dtype)):
                if allow_numeric_types:
                    return float(value1) == float(value2)
                return type(value1) == type(value2) and value1 == value2
        
        # Handle string types
        if allow_numpy_string and isinstance(value1, (str, np.str_)) and isinstance(value2, (str, np.str_)):
            return str(value1) == str(value2)
        
        # Handle Inf comparisons
        if allow_numpy_inf and (value1 in {float('inf'), float('-inf'), np.inf, -np.inf}) and \
           (value2 in {float('inf'), float('-inf'), np.inf, -np.inf}):
            return allow_equal_inf or (value1 == value2)
        
        # Handle datetime types
        if allow_numpy_datetime or allow_pandas_datetime:
            if isinstance(value1, (datetime, np.datetime64, pd.Timestamp)) and \
               isinstance(value2, (datetime, np.datetime64, pd.Timestamp)):
                return pd.Timestamp(value1) == pd.Timestamp(value2)
            if isinstance(value1, (timedelta, np.timedelta64, pd.Timedelta)) and \
               isinstance(value2, (timedelta, np.timedelta64, pd.Timedelta)):
                return pd.Timedelta(value1) == pd.Timedelta(value2)
            if isinstance(value1, pd.Period) or isinstance(value2, pd.Period):
                return isinstance(value1, pd.Period) and isinstance(value2, pd.Period) and value1 == value2
        
        # Handle sequence/mapping types (not yet implemented, placeholders for future enhancements)
        if isinstance(value1, (list, tuple, range, deque, np.ndarray, pd.Series, pd.Index)) and \
           isinstance(value2, (list, tuple, range, deque, np.ndarray, pd.Series, pd.Index)):
            return NotImplemented  # Placeholder: More complex logic to be added later
        
        if isinstance(value1, (dict, OrderedDict)) and isinstance(value2, (dict, OrderedDict)):
            return NotImplemented  # Placeholder: More complex logic to be added later
        
        if isinstance(value1, (set, frozenset)) and isinstance(value2, (set, frozenset)):
            return NotImplemented  # Placeholder: More complex logic to be added later
        
        # Default Python behavior for all other cases
        return value1 == value2

    def _is_iterable(self, value):
        """Returns True if value is iterable, otherwise False."""
        try:
            iter(value)
            return True
        except TypeError:
            return False

    def __repr__(self) -> str:
        """
        Returns a developer-friendly string representation of the object.

        This representation is useful for debugging and introspection.

        Returns
        -------
        str
            A string that represents the class in a way that includes its module name.
        """
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """
        Returns a user-friendly string representation of the object.

        This is useful when printing the object for human-readable output.

        Returns
        -------
        str
            A simple string representation indicating that this is a DataOperations instance.
        """
        return "DataOperations: A utility class for handling data operations."

