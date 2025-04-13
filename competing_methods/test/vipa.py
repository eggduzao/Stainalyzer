"""
====================================================================================================
yaml.py - General Module for YAML Processing, Analysis, and Support
====================================================================================================

Overview
--------
This module, 'yaml.py', serves as a centralized collection of utility functions and classes designed 
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
```

**Example 2: Data Validation with Rule Engine**

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
from pathlib import Path
from collections import OrderedDict, Counter
from typing import Generator, List, Any, Callable, Dict, Tuple, Optional

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

class UnifiedLoader(yaml.SafeLoader):
    """
    Custom YAML Loader that supports environment variables and custom types like `!positive`.
    """

    def __init__(self, stream):
        super().__init__(stream)
        self._add_custom_constructors()

    def _add_custom_constructors(self):
        """
        Registers all custom YAML constructors for special tags like `!positive`.
        """
        self.add_constructor("!env", self._env_var_constructor)
        self.add_constructor("!positive", self._positive_constructor)

    def _env_var_constructor(self, loader, node):
        """
        Resolves environment variables in YAML.
        Example: key: !env $HOME

        Parameters
        ----------
        loader : yaml.SafeLoader
            The YAML loader instance.
        node : yaml.Node
            The YAML node containing the variable to be expanded.

        Returns
        -------
        str
            The expanded environment variable string.

        Example
        -------
        host: "${SERVER_HOST}" -> "localhost"
        """
        value = loader.construct_scalar(node)
        return Path(os.path.expandvars(value)).expanduser()

    def _positive_constructor(self, loader, node):
        """
        Ensures the `!positive` tag only accepts numbers greater than zero.
        """
        value = loader.construct_scalar(node)
        try:
            number = float(value)
            if number > 0:
                return number
            else:
                raise ValueError(f"Invalid value for !positive: {value} (must be > 0)")
        except ValueError:
            raise ValueError(f"Invalid value for !positive: {value} (must be a number)")

class Rule:
    """
    A class representing various types of rules for data validation.

    This class supports rule categories like:
    - Boolean Rules: True/False conditions.
    - Numeric Rules: Value comparisons and ranges.
    - String Rules: Length checks and pattern matches.
    - Category Rules: Membership or combinations within a set.

    Attributes
    ----------
    name : str
        The name of the rule.
    condition : str
        The logical condition for the rule.
    rule_type : str
        The category/type of the rule (e.g., "boolean", "numeric", "string", "category").
    parameters : dict
        Additional parameters relevant to the rule type.

    Methods
    -------
    evaluate(value)
        Evaluate the rule against a provided value.
    """

    def __init__(self, name : str, condition : str, rule_type : str, parameters : Dict=None):
        """
        Initialize a Rule object.

        Parameters
        ----------
        name : str
            The name of the rule.
        condition : str
            The logical condition or rule description.
        rule_type : str
            The category of the rule.
        parameters : dict, optional
            Additional parameters needed for evaluation.

        Raises
        ------
        ValueError
            If the rule_type is unsupported.
        """
        supported_types = {"boolean", "numeric", "string", "category", "positive"}
        if rule_type not in supported_types:
            raise ValueError(f"Unsupported rule type: {rule_type}")

        self.name = name
        self.condition = condition
        self.rule_type = rule_type
        self.parameters = parameters if parameters else {}

    def __repr__(self):
        """
        Developer-friendly string representation.

        Returns
        -------
        str
            String representing the rule in detail.
        """
        return (
            f"Rule(name={self.name}, condition={self.condition}, "
            f"rule_type={self.rule_type}, parameters={self.parameters})"
        )

    def __str__(self):
        """
        User-friendly string representation.

        Returns
        -------
        str
            A descriptive string of the rule.
        """
        return f"[{self.rule_type}] Rule '{self.name}': {self.condition}"

    def evaluate(self, value):
        """
        Evaluate the rule against a given value.

        Parameters
        ----------
        value : any
            The value to be checked against the rule.

        Returns
        -------
        bool
            True if the value passes the rule; False otherwise.

        Raises
        ------
        ValueError
            If evaluation logic for the given rule_type is not defined.
        """
        if self.rule_type == "boolean":
            expected = self.parameters.get("expected")
            return bool(value) == expected

        elif self.rule_type == "numeric":
            min_val = self.parameters.get("min")
            max_val = self.parameters.get("max")
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True

        elif self.rule_type == "positive":
            return isinstance(value, (int, float)) and value > 0

        elif self.rule_type == "string":
            min_len = self.parameters.get("min_len")
            max_len = self.parameters.get("max_len")
            if min_len is not None and len(value) < min_len:
                return False
            if max_len is not None and len(value) > max_len:
                return False
            return True

        elif self.rule_type == "category":
            allowed = self.parameters.get("allowed")
            if not allowed:
                raise ValueError("'allowed' parameter required for category rules.")
            return value in allowed

        else:
            raise ValueError(f"Unsupported rule type: {self.rule_type}")

        return super().evaluate(value)  # Other rule types work as usual

class YamlUtils:
    """
    A utility class for handling YAML file operations.

    # Register constructors for YAML parsing
    UnifiedLoader.add_constructor('tag:yaml.org,2002:str', env_var_constructor)
    UnifiedLoader.add_constructor('!rule', lambda loader, node: Rule(**loader.construct_mapping(node)))

    Attributes
    ----------
    input_path : str
        Path to the YAML file to be processed.
    yaml_dict : dict
        Dictionary representation of the YAML file contents.

    Methods
    -------
    __repr__()
        Returns a developer-friendly string representation of the object.
    __str__()
        Returns a user-friendly string representation of the object.
    _read_yaml()
        Reads the YAML file into memory and stores it in self.yaml_dict.
    _yaml_check(self, yaml_file: str, yaml_key: str, yaml_values: list):
    """

    def __init__(self, input_path : Path = None):
        """
        Initialize the YamlUtils class with an optional YAML file path.

        Parameters
        ----------
        input_path : str, optional
            Path to the YAML file to be processed. Default is None.

        Attributes
        ----------
        input_path : str
            Path to the YAML file to be processed. Default is None.
        yaml_dict : Dict[Any, Any]
            Placeholder Default is None.
        """
        self.input_path = input_path
        self.yaml_dict = {}
            # self.rules = []
        if self.input_path:
            try:
                self._read_yaml()
                inconsistent_lines = self._yaml_check("one_hot_cluster", ["name", "eligible"])
                if inconsistent_lines:
                    message = f"Inconsistent occurrences found at lines: {result}"
                    file_path = str(self.input_path)
                    rule_name = "YAMLFieldConsistency"
                    conflicting_values = f"{inconsistent_lines}"
                    raise YAMLValidationError(message, file_path, rule_name, conflicting_values, severity="error")
            except YAMLValidationError as e:
                print(f"Error: {e.message}")
                if e.severity == "error":
                    raise
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

    def _read_yaml(self):
        """
        Reads the content of the YAML file and stores it in self.yaml_dict.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        yaml.YAMLError
            If there is an issue parsing the YAML file.
        """
        
        try:
            if not self.input_path.exists() or not self.input_path.is_file() or self.input_path.is_dir():
                raise FileNotFoundError(f"File not found: {self.input_path}")
            with open(self.input_path, 'r') as f:
                self.yaml_dict = yaml.load(f, Loader=UnifiedLoader)  # Use our custom loader
        except FileNotFoundError as e:
            raise e
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse YAML file: {e}")

    # def _extract_rules(self):
    #     """
    #     Automatically extracts rules from the parsed YAML.
    #     """
    #     for field, config in self.yaml_dict.items():
    #         if "rule_type" in config:
    #             rule = Rule(
    #                 name=field,
    #                 condition=config.get("condition", ""),
    #                 rule_type=config["rule_type"],
    #                 parameters=config.get("parameters", {})
    #             )
    #             self.rules.append(rule)

    # def get_rules(self):
    #     """
    #     Returns a list of extracted Rule objects.
    #     """
    #     return self.rules

    def _yaml_check(self, yaml_key: str, yaml_values: list):
        """
        Reads the content of the YAML file and checks for consistency.

        Raises
        ------
        Placeholder

        Returns
        -------
        Placeholder
        """

        inconsistent_lines = self._check_yaml_field_consistency(self.input_path, yaml_key, yaml_values)

        return inconsistent_lines  # List of line numbers where inconsistency was found

    def _check_yaml_field_consistency(self, yaml_file: Path, yaml_key: str, yaml_values: list):
        """
        Recursively walks through all levels of a YAML dictionary (parsed from a file) and checks 
        if a specified key ('yaml_key') contains all required keys ('yaml_values').

        If the 'yaml_key' exists but does NOT contain all 'yaml_values', it reports 
        the line number in the YAML file where this happened.

        Args:
            yaml_file (Path): The path to the YAML file.
            yaml_key (str): The key to look for.
            yaml_values (list): The list of expected keys inside the 'yaml_key' dictionary.

        Returns:
            list: A list of line numbers where 'yaml_key' was missing required values.
        """
        
        # Load YAML file with line number tracking
        with open(yaml_file, 'r') as file:
            yaml_content = file.read()
            yaml_dict = yaml.safe_load(yaml_content)

        # To track inconsistent entries
        inconsistent_lines = []

        def _recursive_check(sub_dict, path=""):
            """Recursive function to navigate through nested dictionaries."""
            if isinstance(sub_dict, dict):
                for key, value in sub_dict.items():
                    new_path = f"{path}.{key}" if path else key  # Track dictionary path

                    # If the key matches, check if it contains all required values
                    if key == yaml_key and isinstance(value, dict):
                        missing_values = [val for val in yaml_values if val not in value]
                        
                        if missing_values:
                            # Locate the line number where the key appears
                            for i, line in enumerate(yaml_content.splitlines(), start=1):
                                if f"{yaml_key}:" in line:
                                    inconsistent_lines.append(i)
                                    break  # Capture first occurrence only

                    # Recurse into nested dictionaries
                    if isinstance(value, dict):
                        _recursive_check(value, new_path)

        _recursive_check(yaml_dict)

        return inconsistent_lines  # List of line numbers where inconsistency was found

    def print_tree(self, data: Any, vector: List, indent: str="", is_last: bool=True):
        """
        Recursively prints a nested dictionary (YAML structure) in a tree-like format,
        displaying key-value pairs correctly.

        Args:
            data (dict): The dictionary parsed from YAML.
            indent (str): The indentation level for formatting.
            is_last (bool): Whether the current item is the last child (affects tree visuals).
        """
        if isinstance(data, dict):

            keys = list(data.keys())
            
            for i, key in enumerate(keys):

                is_last_key = i == len(keys) - 1
                connector = "└── " if is_last_key else "├── "
                vector.append(f"{indent}{connector}{key}:\n")

                # Recursively call function with increased indentation
                new_indent = indent + ("    " if is_last_key else "│   ")
                self.print_tree(data[key], vector, new_indent, is_last_key)

        elif isinstance(data, list):

            for i, item in enumerate(data):
            
                is_last_item = i == len(data) - 1
                connector = "└── " if is_last_item else "├── "
                vector.append(f"{indent}{connector}{item}:\n")
                new_indent = indent + ("    " if is_last_item else "│   ")
                self.print_tree(item, vector, new_indent, is_last_item)

        else:
            
            vector.append(f"{indent}└── {repr(data)}\n")  # Print leaf values formatted properly

    def save_yaml(self,
                  yaml_dict: Dict[Any, Any], 
                  output_file: Path, 
                  sort_keys: bool=False, 
                  default_flow_style: bool=False, 
                  allow_unicode: bool=True) -> None:
        with open(output_file, 'w', encoding='utf-8') as file:
            yaml.dump(yaml_dict,
                      file,
                      sort_keys = sort_keys,
                      default_flow_style = default_flow_style,
                      allow_unicode = allow_unicode)

    def __repr__(self):
        """
        Returns a developer-friendly string representation of the object.

        Returns
        -------
        str
            Developer-friendly representation of the object.
        """
        return f"YamlUtils(input_path={self.input_path!r})"

    def __str__(self):
        """
        Returns a user-friendly string representation of the object.

        Returns
        -------
        str
            User-friendly representation of the object.
        """
        return f"YAML Utils for file: {self.input_path}"

