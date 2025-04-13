"""
====================================================================================================
main.py - General Module for I/O Processing, Analysis, and Support
====================================================================================================

Overview
--------
This module, 'main.py', placeholder.


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


Development Notes:
-------------

 - Python Compatibility: Python 3.10 and above.

 - Required Packages: NumPy, Pandas, logging.
 
 - Testing Framework: The module includes unit tests implemented using pytest to ensure reliability
across different Python environments.
 
 - Code Style Compliance: All code follows PEP 8 guidelines, with additional comments to aid
maintainability for future developers.

Usage Example:
--------------

```python
Placeholder
```

Author: Eduardo Gade Gusmao
Created On: 11/12/2024
Last Updated: 15/02/2025
Version: 0.1.0
License: <Currently_Withheld>

"""

from .core import core_function, parse_args

def main():
    """
    Main entry point for the Stainalyzer tool.

    This function parses command-line arguments and initiates the core processing
    by calling `core_function` with the provided arguments.

    """

    # Parse arguments
    args = parse_args()

    # Call the core function with parsed arguments
    core_function(
        input_training_folder=args.input_training_folder,
        output_training_folder=args.output_training_folder,
        root_name=args.root_name,
        training_severity=args.severity
    )
