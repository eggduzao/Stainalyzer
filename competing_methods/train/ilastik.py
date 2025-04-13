
"""
metrics
-------
This module contains a QualityMetric Interface and a class to implement each of the
different Quality Dimention metrics.

CompletenessMetric
ConformanceMetric
ConsistencyMetric
AgreementMetric
RelevanceMetric
RepresentativenessMetric
ContextualizationMetric

### QualityMetric Example

class CompletenessMetric(QualityMetric):
    def calculate(self) -> np.float64:
        pass
    def case(self) -> Generator[Any, None, None]:
        pass
    def report(self) -> List[Any]:
        pass



---

### ConformanceMetric Example

def validate_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0

rules = {
    "age": validate_positive,
    "height": lambda x: isinstance(x, (int, float)) and 0.5 <= x <= 2.5,  # in meters
    "gender": lambda x: x in {"male", "female", "other"}
}

---

### ConsistencyMetric Example

def check_age_vs_birth_year(data: Any) -> bool:
    # Example: Ensure 'age' aligns with 'birth_year' in the dataset.
    for row in data:
        if row["birth_year"] + row["age"] != 2023:  # Replace with current year
            return False
    return True

rules = [
    ("Age matches birth year", check_age_vs_birth_year),
    ("Height vs weight correlation", lambda data: check_height_weight(data)),
]

---

### AgreementMetric Example

def rater_agreement(data: Any, rater1: str, rater2: str) -> bool:
    "" "
    Checks agreement between two raters for all data points.

    Parameters:
    ----------
    data : Any
        The dataset containing rater columns.
    rater1 : str
        The column name for the first rater.
    rater2 : str
        The column name for the second rater.

    Returns:
    -------
    bool
        True if values agree, False otherwise.
    "" "
    for row in data:
        if row[rater1] != row[rater2]:
            return False
    return True

comparisons = [
    {"Rater1 vs Rater2": lambda data: rater_agreement(data, "Rater1", "Rater2")},
    {"SourceA vs SourceB": lambda data: rater_agreement(data, "SourceA", "SourceB")},
]

---

### RelevanceMetric Example

def is_relevant_age(data: Any) -> bool:
    "" "
    Checks if the age data is within a relevant range (e.g., for a clinical study).

    Parameters:
    ----------
    data : Any
        A single data point or row containing an 'age' field.

    Returns:
    -------
    bool
        True if the age is relevant, False otherwise.
    "" "
    return 18 <= data["age"] <= 65

def is_relevant_condition(data: Any) -> bool:
    "" "
    Checks if the medical condition is relevant for the study.

    Parameters:
    ----------
    data : Any
        A single data point or row containing a 'condition' field.

    Returns:
    -------
    bool
        True if the condition is relevant, False otherwise.
    "" "
    return data["condition"] in {"diabetes", "hypertension"}

criteria = [is_relevant_age, is_relevant_condition]

---

### RepresentativenessMetric Example

def check_age_distribution(data: Any) -> bool:
    "" "
    Validates whether the age distribution matches the target population.

    Parameters:
    ----------
    data : Any
        The dataset containing an 'age' column.

    Returns:
    -------
    bool
        True if the age distribution is representative, False otherwise.
    "" "
    # Example: Check if 20% of the dataset consists of ages 18-25
    age_18_25 = sum(18 <= row["age"] <= 25 for row in data) / len(data)
    return 0.15 <= age_18_25 <= 0.25

def check_gender_balance(data: Any) -> bool:
    "" "
    Validates whether the gender distribution matches the target population.

    Parameters:
    ----------
    data : Any
        The dataset containing a 'gender' column.

    Returns:
    -------
    bool
        True if the gender distribution is balanced, False otherwise.
    "" "
    gender_counts = {row["gender"]: gender_counts.get(row["gender"], 0) + 1 for row in data}
    male_percentage = gender_counts.get("male", 0) / len(data)
    return 0.45 <= male_percentage <= 0.55

benchmarks = {
    "Age distribution": check_age_distribution,
    "Gender balance": check_gender_balance,
}

---

### ContextualizationMetric Example

def validate_seasonal_variation(data: Any) -> bool:
    "" "
    Validates whether the data aligns with expected seasonal variations.

    Parameters:
    ----------
    data : Any
        The dataset containing a 'month' and 'value' column.

    Returns:
    -------
    bool
        True if the data aligns with seasonal expectations, False otherwise.
    "" "
    for row in data:
        if row["month"] in {"June", "July", "August"} and row["value"] < 20:
            return False  # Example: Expect higher values in summer months
    return True

def validate_geographic_dependency(data: Any) -> bool:
    "" "
    Validates whether data aligns with geographic dependencies.

    Parameters:
    ----------
    data : Any
        The dataset containing a 'region' and 'measure' column.

    Returns:
    -------
    bool
        True if the data aligns with geographic expectations, False otherwise.
    "" "
    for row in data:
        if row["region"] == "Coastal" and row["measure"] < 50:
            return False  # Example: Coastal areas expect higher measures
    return True

context_rules = {
    "Seasonal variation": validate_seasonal_variation,
    "Geographic dependency": validate_geographic_dependency,
}

"""

"""
====================================================================================================
metrics.py - General Module for I/O Processing, Analysis, and Support
====================================================================================================

Overview
--------
This module, 'metrics.py', placeholder.


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
import yaml
import copy
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import OrderedDict, Counter
from typing import Generator, List, Any, Callable, Dict, Tuple, Optional

from .exceptions import (
    DimensionMetricException,
    #-------------------------------------
    CompletenessError,
    MissingDataError,
    DuplicateDataError,
    #-------------------------------------
    ConformanceError,
    IncorrectDataFormatError,
    InvalidDataTypeError,
    InvalidDataValueError,
    InvalidDataUnitError,
    #-------------------------------------
    ConsistencyError,
    InconsistentDataError,
    #-------------------------------------
    AgreementError,
    IncorrectLOINCStandard,
    IncorrectICDStandard,
    IncorrectHPOStandard,
    IncorrectHL7FHIRStandard,
    #-------------------------------------
    RelevanceError,
    RelevanceRuleIncongruence,
    #-------------------------------------
    RepresentativenessError,
    EntropyCalculationError,
    RepresentativenessRuleIncongruence,
    #-------------------------------------
    ContextualizationError,
    ContextualizationRuleIncongruence,
    #-------------------------------------
    YAMLConfigurationException,
    YAMLFileNotFoundError,
    YAMLFormatError,
    YAMLValidationError,
    #-------------------------------------
    OutputCreationException,
    ReportGenerationError,
    ErrorYieldException,
    MetricComputationError,
    PlotInvalidDataError,
    PlotComputationError,
)
from .utils import PandasUtils
from .yaml import YamlUtils
from .data import DataOperations

###############################################################################
# Constants
###############################################################################

# Constants
SEED = 1987
np.random.seed(SEED)

###############################################################################
# Classes
###############################################################################

class QualityMetric(ABC):
    """
    Abstract base class (interface) for quality metrics.

    This class provides a scaffold for specific metric implementations, ensuring
    consistent structure and behavior across all quality metric classes.
    """

    def __init__(self, data: Any):
        """
        Initializes the QualityMetric with the provided dataset.

        Parameters
        ----------
        data : Any
            The dataset or structure on which the metric operates.
        """
        self.data = data
        self._pandasutils = PandasUtils()
        self._dataop = DataOperations()

    @abstractmethod
    def calculate(self) -> np.float64:
        """
        Calculates the quality metric score.

        Returns
        -------
        np.float64
            A score indicating the quality dimension value.
        """
        pass

    @abstractmethod
    def case(self) -> Generator[Any, None, None]:
        """
        Generates individual cases that contribute to a lower score.

        Yields
        ------
        Any
            Elements that negatively impact the quality metric score.
        """
        pass

    @abstractmethod
    def report(self) -> List[Any]:
        """
        Produces a comprehensive report of all elements contributing to a lower score.

        Returns
        -------
        List[Any]
            A list of elements that negatively impact the score.
        """
        pass

    def validate_data(self) -> bool:
        """
        Validates the input data to ensure compatibility with the metric.

        Returns
        -------
        bool
            True if the data is valid, False otherwise.

        Raises
        ------
        ValueError
            If the data is invalid (e.g., None or unsupported format).
        """
        if self.data is None:
            raise ValueError("Data cannot be None.")
        return True

    def normalize_score(self, score: np.float64) -> np.float64:
        """
        Normalizes a raw score to a predefined range, if necessary.

        Parameters
        ----------
        score : np.float64
            The raw score to normalize.

        Returns
        -------
        np.float64
            A normalized score clamped between 0 and 1.
        """
        return max(0.0, min(1.0, score))

    @abstractmethod
    def description(self) -> str:
        """
        Provides a description of the metric, including its purpose and methodology.

        Returns
        -------
        str
            A string describing the metric and its role.
        """
        return f"{self.__class__.__name__}: A quality metric implementation."

class CompletenessMetric(QualityMetric):
    """
    Measures the completeness and uniqueness of the data.

    This class calculates completeness by identifying missing values and checking for duplicate rows.
    It supports bagging-based completeness/uniqueness estimation and applies customizable rules from YAML configuration.

    Exceptions:
        CompletenessError: Raised for general completeness metric issues.
        MissingDataError: Raised when expected data fields are missing.
        DuplicateDataError: Raised when duplicate entries are found.

    Attributes
    ----------
    data : pd.DataFrame
        The dataset to analyze.
    missing_values : List[Any]
        List of values considered as missing.
    global_yaml : YamlUtils
        YAML object containing global metric settings.
    completeness_yaml : YamlUtils
        YAML object with table-specific completeness rules.
    yaml_rules : Dict[Any, Any]
        Placeholder
    _missing_locations : List[Tuple[int, str]]
        Records of missing data locations.
    _missing_report : List[Tuple]
        Placeholder
    _duplicate_rows : List[int]
        Records of indices for duplicated rows.
    _duplicate_report : List[Tuple]
        Placeholder
    _pandasutils : PandasUtils Object
        Placeholder
    """

    def __init__(self,
                 data: pd.DataFrame,
                 global_yaml: YamlUtils = None,
                 completeness_yaml: YamlUtils = None,
                 any_flag: bool = True):

        super().__init__(pd.DataFrame(data))
        self.missing_values = [None, np.nan, "NA", "N/A"] # Default missing values
        self.global_yaml = global_yaml
        self.completeness_yaml = completeness_yaml
        self.yaml_rules = OrderedDict()
        self._missing_locations = []
        self._missing_report = OrderedDict()
        self._duplicate_rows = []
        self._duplicate_report = OrderedDict()

        # Fetching the fields within the dictionary
        if self.completeness_yaml is not None:
            self.yaml_rules = self.completeness_yaml.yaml_dict['completeness_metric']['table_fields']

        # Slicing the data based on the YAML fields
        if self.data is not None:
            self.data = self._pandasutils.slice_with_dictionary(self.data, self.yaml_rules)
            self.data, self.yaml_rules = self._collapse_one_hot(any_flag=any_flag)

    def calculate(self) -> NDArray[np.float32]:
        """
        Calculates the quality metric score.

        Returns
        -------
        np.float64
            A score indicating the quality dimension value.
        """
        missing_count, total_missing_count, completeness = self.calculate_completeness()
        duplicate_count, total_duplicate_count, uniqueness = self.calculate_uniqueness()
        return_list = np.array([missing_count, total_missing_count, completeness,
                                duplicate_count, total_duplicate_count, uniqueness],
                                dtype=np.float32)
        return return_list

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    # COMPLETENESS MAIN:

    def calculate_completeness(self) -> Dict[str, np.float32]:
        """
        Calculate the completeness score for the dataset.

        Completeness is defined as the proportion of non-missing values relative to the total fields.

        Returns
        -------
        NDArray[np.float32]
            Array with [missing_count, total_count, completeness_score].

        Raises
        ------
        MissingDataError
            If a required field is missing.
        CompletenessError
            For generic completeness issues.
        """

        try:
            results = self._calculate_missing(self.data, create_missing_structure=True, create_missing_report=True)

            # Ensure the function returned exactly 3 values
            if not isinstance(results, (list, tuple)) or len(results) != 3:
                raise CompletenessError("Function `_calculate_missing` did not return exactly 3 values.")

            # Create dictionary
            name_vector = ["Total Missing Values: ", "Total Table Values: ", "Completeness: "]
            result_vector = [np.float32(e) if isinstance(e, float) else np.int16(e) for e in results]
            full_result = {name: result for name, result in zip(name_vector, result_vector)}

            # Convert result to NumPy array
            return full_result

        except CompletenessError:  
            # If we already raised CompletenessError, let it propagate
            raise  
        except Exception as e:  
            # Any other error, wrap in CompletenessError
            raise CompletenessError("Error calculating completeness") from e

    def calculate_bagging_completeness(self) -> Dict[str, np.float32]:
        """
        Calculate completeness using bagging.

        Bagging involves repeated sampling of rows and columns with replacement to estimate variance.

        Returns
        -------
        NDArray[np.float32]
            Array with concatenated results for row-based and column-based completeness.

        Raises
        ------
        CompletenessError
            If bagging calculation fails.
        """

        n_rows, n_cols = self.data.shape
        bagging_size_row = n_rows
        bagging_size_col = n_cols
        bagging_times = self.global_yaml.yaml_dict['completeness_metric']['bagging_times']

        row_result_vector = []
        col_result_vector = []
        random_row_result_vector = []
        random_col_result_vector = []

        # Create mock random completeness dataset
        try:
            resvec = self._calculate_missing(self.data)
            # Ensure the function returned exactly 3 values
            if not isinstance(resvec, (list, tuple)) or len(resvec) != 3:
                raise CompletenessError("Function `_calculate_missing` did not return exactly 3 values.")
            random_data = self._generate_random_completeness(self.data, resvec[-1])
        except CompletenessError:  
            # Propagation
            raise  
        except Exception as e:  
            # Any other error, wrap in CompletenessError
            raise CompletenessError("Error calculating completeness") from e

        # Generate bagging results
        try:

            for i in range(bagging_times):

                sampled_rows = self.data.sample(n=bagging_size_row, replace=True)
                sampled_cols = self.data.sample(n=bagging_size_col, axis=1, replace=True)
                random_rows = random_data.sample(n=bagging_size_row, replace=True)
                random_cols = random_data.sample(n=bagging_size_col, axis=1, replace=True)

                row_result = self._calculate_missing(sampled_rows)
                # Ensure the function returned exactly 3 values
                if not isinstance(row_result, (list, tuple)) or len(row_result) != 3:
                    raise CompletenessError("Function `_calculate_missing` did not return exactly 3 values.")

                row_result_vector.append(row_result[-1])

                col_result = self._calculate_missing(sampled_cols)
                # Ensure the function returned exactly 3 values
                if not isinstance(col_result_vector, (list, tuple)) or len(col_result_vector) != 3:
                    raise CompletenessError("Function `_calculate_missing` did not return exactly 3 values.")

                col_result_vector.append(col_result[-1])

                random_row_result = self._calculate_missing(random_rows)
                # Ensure the function returned exactly 3 values
                if not isinstance(random_row_result, (list, tuple)) or len(random_row_result) != 3:
                    raise CompletenessError("Function `_calculate_missing` did not return exactly 3 values.")

                random_row_result_vector.append(random_row_result[-1])

                random_col_result = self._calculate_missing(random_cols)
                # Ensure the function returned exactly 3 values
                if not isinstance(random_col_result, (list, tuple)) or len(random_col_result) != 3:
                    raise CompletenessError("Function `_calculate_missing` did not return exactly 3 values.")

                random_col_result_vector.append(random_col_result[-1])

            # Row and column numpy arrays
            row_result_vector = np.array(row_result_vector, dtype=np.float32)
            col_result_vector = np.array(col_result_vector, dtype=np.float32)
            random_row_result_vector = np.array(random_row_result_vector, dtype=np.float32)
            random_col_result_vector = np.array(random_col_result_vector, dtype=np.float32)

            # Mean and std vector
            result_vector = np.array([ np.mean(row_result_vector),
                                       np.std(row_result_vector, ddof=1),
                                       np.mean(col_result_vector),
                                       np.std(col_result_vector, ddof=1),
                                       np.mean(random_row_result_vector),
                                       np.std(random_row_result_vector, ddof=1),
                                       np.mean(random_col_result_vector),
                                       np.std(random_col_result_vector, ddof=1),
                                       ], dtype=np.float32)

            # Name vector
            name_vector = ["Row Completeness Mean",
                           "Row Completeness Std",
                           "Column Completeness Mean",
                           "Column Completeness Std",
                           "Random Row Completeness Mean",
                           "Random Row Completeness Std",
                           "Random Column Completeness Mean",
                           "Random Column Completeness Std",
                           ]

            # Create dictionary with zip
            full_result = {name: result for name, result in zip(name_vector, result_vector)}

            return full_result

        except CompletenessError:  
            # Propagation
            raise  
        except Exception as e:  
            # Any other error, wrap in CompletenessError
            raise CompletenessError("Error calculating completeness") from e

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    # MISSING:

    def _calculate_missing(self,
                           data : pd.DataFrame, 
                           create_missing_structure: bool = False,
                           create_missing_report: bool = False) -> tuple:
        """
        Core method for detecting missing values.

        Missing values are identified based on predefined rules and specified missing patterns.

        Parameters
        ----------
        data : pd.DataFrame
            The data to check.
        rules : Dict[Any, Any]
            Dictionary with rules and parameters to check for missing data.
        create_missing_structure : bool
            If True, records missing value locations.

        Returns
        -------
        tuple
            Contains (missing_count, total_count, completeness_score).

        Raises
        ------
        MissingDataError
            If missing data is found.
        """

        missing_count = 0
        total_missing_count = 0
        all_fields = data.columns.to_list()

        # Iterating on rows (patients)
        for idx, row in data.iterrows():

            # Iterating on the columns (fields)
            for field, value in row.items():

                # Implmenentation of YAMLFieldNotFound 
                try:
                    if field not in all_fields:
                        raise YAMLFieldNotFound(
                            message=f"Field '{field}' not found in YAML configuration.",
                            file_path=".this.rules.completeness.yaml",
                            rule_name="YAMLFieldExistence",
                            conflicting_table_field=field,
                            severity='error'
                        )
                except YAMLFieldNotFound as e:
                    print(f"Warning: {e.message}")
                    if e.severity == 'error':
                        raise

                # Check missing:
                is_missing, status, missing_weight, log_message = self._calculate_missing_instance(data, idx, field)

                # Data is missing
                if is_missing:

                    # Count missing data
                    missing_count += missing_weight
                    
                    # Missing Structure
                    if create_missing_structure:
                        self._missing_locations.append((idx, field, status, missing_weight))

                    # Missing Report
                    if create_missing_report:
                        if idx not in self._missing_report:
                            self._missing_report[idx] = OrderedDict()
                        if field not in self._missing_report[idx]:
                            self._missing_report[idx][field] = OrderedDict()
                        if field in self.yaml_rules:
                            result_list = [idx, field, value, is_missing, status, missing_weight, 
                                           self.yaml_rules[field], log_message]
                            self._missing_report[idx][field] = result_list

                total_missing_count += missing_weight

        completeness_score = 1.0 - (missing_count / total_missing_count) if total_missing_count else 1.0

        return missing_count, total_missing_count, completeness_score

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    # MISSING HELPERS:

    def _calculate_missing_instance(self,
                                    data : pd.DataFrame,
                                    row_idx : int,
                                    field : str) -> bool:
        """
        Check if instance is missing value
        """

        # USE: self._dataop.compare(value, value)
        # USE: self._dataop.is_in(value, value)
        # USE: self._dataop.is_not_in(value, value)

        # Function's unique ID
        unique_id = str(np.random.randint(0, 999)).zfill(3)

        # Initializing log list
        log_message = []

        # Logging #############################################################
        log1 = f"1. ENTERING _calculate_missing_instance() with UID = {unique_id}:"
        log2 = f"            - row_idx = {row_idx}"
        log3 = f"            - field = {field}"
        for log in [log1, log2, log3]:
            log_message.append(log)
        #######################################################################

        # Initialization
        current_rules = self.yaml_rules[field]
        missing_weight = current_rules["missing_weight"] # Get missing weight
        value = data.loc[row_idx, field]  # Get the value

        if isinstance(value, pd.Series):
            if len(value) > 1:

                # Logging #############################################################
                log1 = f"W. WARNING in _calculate_missing_instance() with UID = {unique_id}:"
                log2 = f"   WARNING: More than one entry was found on ({row_idx}, {field})."
                log3 = f"            -> Entries = {value}"
                log4 = f"            -> Using only = {value.iloc[0]}"
                for log in [log1, log2, log3, log4]:
                    log_message.append(log)
                #######################################################################

            value = value.iloc[0]  # Extract first value if multiple rows exist
        is_na = pd.isna(value)

        # Check additional N/A values
        missing_values = []
        if "na_criteria" in current_rules:
            for rule_value in current_rules.get("na_criteria"):
                if self._dataop.compare(rule_value, None):
                    missing_values.append(self._dataop.compare(value, None))
                    missing_values.append(self._dataop.is_in(value, self.missing_values))
                    missing_values.append(is_na)
                else:
                    missing_values.append(self._dataop.compare(value, rule_value))

        # Checking missing condition
        missing_condition = not any(missing_values)

        # Logging #############################################################
        log1 = f"2. _calculate_missing_instance() with UID = {unique_id}:"
        log2 = f"   - missing_values = {missing_values}"
        lvec=[]
        if not any(missing_values):
            log3 = f" 2. EXITING _calculate_missing_instance() with UID = {unique_id}:"
            log4 = f"            - RETURNING: Missing Value = FALSE."
            log5 = f"            - No missing values in the current cell."
            lvec = [log1, log2, log3, log4, log5]
        else:
            log3 = f"   - Continue, since there are missing values."
            lvec = [log1, log2, log3]
        for log in lvec:
            log_message.append(log)
        #######################################################################

        # If has at least one of the NA eligibility conditions
        if missing_condition:
            return (False, "1", missing_weight, log_message)

        # Logging #############################################################
        log1 = f"3. _calculate_missing_instance() with UID = {unique_id}:"
        log2 = f"   - current_rules = {current_rules}"
        if "conditional_fields" in current_rules and current_rules["conditional_fields"] is not None:
            log3 = f"   - Processing conditional values in this field." 
        else:
            log3 = f"   - No conditional values found. Jumping to 5."      
        for log in [log1, log2, log3]:
            log_message.append(log)
        #######################################################################

        # Conditional field
        if "conditional_fields" in current_rules and current_rules["conditional_fields"] is not None:

            is_eligible = self._check_eligibility(data, row_idx, field,
                                                  current_rules["conditional_fields"], log_message)

            # Logging #############################################################
            log1 = f"4. _calculate_missing_instance() with UID = {unique_id}:"
            log2 = f"   - Conditional fields processed successfully with _check_eligibility()."
            lvec=[]
            if not is_eligible:
                log3 = f"4. EXITING _calculate_missing_instance() with UID = {unique_id}:"
                log4 = f"           - RETURNING: Missing Value = FALSE."
                log5 = f"           - Missing value is not eligible."
                lvec = [log1, log2, log3, log4, log5]
            else:
                log3 = f"   - Field is eligible to be considered missing."
                lvec = [log1, log2, log3]
            for log in lvec:
                log_message.append(log)
            #######################################################################

            if not is_eligible:
                return (False, "2", missing_weight, log_message)

        # Logging #############################################################
        log1 = f"5. EXITING _calculate_missing_instance() with UID = {unique_id}:"
        log2 = f"           - RETURNING:"
        log3 = f"           - Missing Value = TRUE."
        for log in [log1, log2, log3]:
            log_message.append(log)
        #######################################################################

        # Return missing
        return (True, "3", missing_weight, log_message)

    def _check_eligibility(self, 
                           data: pd.DataFrame,
                           row_idx: int,
                           field: str,
                           current_rules: dict,
                           log_message: list) -> bool:
        """Determines if a field is eligible based on conditional dependencies."""

        # Function's unique ID
        unique_id = str(np.random.randint(0, 999)).zfill(3)

        # Logging #############################################################
        log1 = f"## 1. ENTERING _check_eligibility() with UID = {unique_id}:"
        log2 = f"               - row_idx = {row_idx}"
        log3 = f"               - field = {field}"
        log4 = f"               - current_rules = {current_rules}"
        for log in [log1, log2, log3, log4]:
            log_message.append(log)
        #######################################################################

        # Initialize lists for storing results
        result_or_list = []
        result_and_list = []
        final_operation = "OR" if "OR" in current_rules else "AND"

        # Logging #############################################################
        log1 = f"## 2. _check_eligibility() with UID = {unique_id}:"
        log2 = f"      - Prepared final_operation = {final_operation}"
        log3 = f"      - Will now execute _lookup_eligibility()."
        for log in [log1, log2, log3]:
            log_message.append(log)
        #######################################################################

        # Recursively check dependencies
        self._lookup_eligibility(data, row_idx, field, current_rules[final_operation], final_operation,
                                 result_or_list, result_and_list, log_message)

        # Logging #############################################################
        log1 = f"## 3. _check_eligibility() with UID = {unique_id}:"
        log3 = f"      - _lookup_eligibility() executed successfully."
        log3 = f"      - Returned: result_or_list = {result_or_list}"
        log4 = f"      - Returned: result_and_list = {result_and_list}"
        for log in [log1, log2, log3, log4]:
            log_message.append(log)
        #######################################################################

        # Compute final eligibility
        result_list = []
        if result_or_list and result_or_list is not None:
            result_list.append(any(result_or_list))
        if result_and_list and result_and_list is not None:
            result_list.append(all(result_and_list))

        final_eligibility = any(result_list) if final_operation == "OR" else all(result_list)

        # Logging #############################################################
        log1 = f"## 4. EXITING _check_eligibility() with UID = {unique_id}:"
        log2 = f"           - RETURNING:"
        log3 = f"           - final_eligibility = {final_eligibility}"
        for log in [log1, log2, log3]:
            log_message.append(log)
        #######################################################################

        return final_eligibility

    # Function _check_eligibility - The hard recursion function:
    def _lookup_eligibility(self, 
                            data: pd.DataFrame,  
                            row_idx: int, 
                            field_name: str,
                            current_rules: dict,
                            operation: str,
                            result_or_list: list,
                            result_and_list: list,
                            log_message: list) -> None:
        """Recurssive Function to check Dependent Fields"""

        # Function's unique ID
        unique_id = str(np.random.randint(0, 999)).zfill(3)

        # Logging #############################################################
        log1 = f"#### 1. ENTERING _lookup_eligibility() with UID = {unique_id}:"
        log2 = f"                 - row_idx = {row_idx}"
        log3 = f"                 - field_name = {field_name}"
        log4 = f"                 - operation = {operation}"
        log5 = f"                 - result_or_list = {result_or_list}"
        log6 = f"                 - result_and_list = {result_and_list}"
        log7 = f"                 - current_rules = {current_rules}"
        for log in [log1, log2, log3, log4, log5, log6, log7]:
            log_message.append(log)
        #######################################################################

        # Main Loop
        for key, value in current_rules.items():

            # Logging #############################################################
            log1 = f"####oo 2. _lookup_eligibility() with UID = {unique_id}:"
            log2 = f"          - Main Loop: key = {key}, value = {value}"
            for log in [log1, log2]:
                log_message.append(log)
            #######################################################################

            # If Key is OR (not lonely field)
            if key == "OR":

                # Logging #############################################################
                log1 = f"####oooo 3. _lookup_eligibility() with UID = {unique_id}:"
                log2 = f"            - Main Loop: key = {key}, value = {value}"
                log3 = f"            - Entered option = OR (running _lookup_eligibility(OR))."
                for log in [log1, log2, log3]:
                    log_message.append(log)
                #######################################################################

                # Create new OR list to get OR rules
                new_or_list = []
                self._lookup_eligibility(data, row_idx, field_name, value, "OR", new_or_list, 
                                         result_and_list, log_message)
                new_or_entry = any(new_or_list) if new_or_list else None

                # Logging #############################################################
                log1 = f"####oooo 4. _lookup_eligibility() with UID = {unique_id}:"
                log2 = f"            - Main Loop: key = {key}, value = {value}"
                log3 = f"            - Entered option = OR (_lookup_eligibility(OR) runned successfuly)."
                log4 = f"            - We have a new_or_list = {new_or_list} (result_and_list = {result_and_list})."
                log5 = f"            - We have a new_or_entry = {new_or_entry}"

                # Append OR's any to main or list
                if new_or_entry is not None:
                    result_or_list.append(new_or_entry)
                    log6 = f"            - The new_or_entry = {new_or_entry} was appended to result_or_list."
                else:
                    log6 = f"            - The new_or_entry = {new_or_entry} was NOT appended to result_or_list."

                for log in [log1, log2, log3, log4, log5, log6]:
                    log_message.append(log)
                #######################################################################

                # Logging #############################################################
                log1 = f"####oooo 5. _lookup_eligibility() with UID = {unique_id}:"
                log2 = f"            - Main Loop: key = {key}, value = {value}"
                log3 = f"            - EXITING current_rules's loop - option OR."
                for log in [log1, log2, log3]:
                    log_message.append(log)
                #######################################################################

            # If Key is AND (not lonely field)
            elif key == "AND":

                # Logging #############################################################
                log1 = f"####oooo 3. _lookup_eligibility() with UID = {unique_id}:"
                log2 = f"            - Main Loop: key = {key}, value = {value}"
                log3 = f"            - Entered option = AND (running _lookup_eligibility(AND))."
                for log in [log1, log2, log3]:
                    log_message.append(log)
                #######################################################################

                # Create new AND list to get AND rules
                new_and_list = []
                self._lookup_eligibility(data, row_idx, field_name, value, "AND", result_or_list, 
                                         new_and_list, log_message)
                new_and_entry = all(new_and_list) if new_and_list else None

                # Logging #############################################################
                log1 = f"####oooo 4. _lookup_eligibility() with UID = {unique_id}:"
                log2 = f"            - Main Loop: key = {key}, value = {value}"
                log3 = f"            - Entered option = AND (_lookup_eligibility(AND) runned successfuly)."
                log4 = f"            - We have a new_and_list = {new_and_list} (result_or_list = {result_or_list})."
                log5 = f"            - We have a new_and_entry = {new_and_entry}"

                # Append AND's all to main and list
                if new_and_entry is not None:
                    result_and_list.append(new_and_entry)
                    log6 = f"            - The new_and_entry = {new_and_entry} was appended to result_and_list."
                else:
                    log6 = f"            - The new_and_entry = {new_and_entry} was NOT appended to result_and_list."

                for log in [log1, log2, log3, log4, log5, log6]:
                    log_message.append(log)
                #######################################################################

                # Logging #############################################################
                log1 = f"####oooo 5. _lookup_eligibility() with UID = {unique_id}:"
                log2 = f"            - Main Loop: key = {key}, value = {value}"
                log3 = f"            - EXITING current_rules's loop - option AND."
                for log in [log1, log2, log3]:
                    log_message.append(log)
                #######################################################################

            # If value is a list = Lonely Field
            elif isinstance(value, list):

                # Try for bootstrapping cases
                try:

                    # Checking field value
                    new_field_name = key
                    new_value = data.at[row_idx, new_field_name]
                    new_current_rules = self.yaml_rules.get(new_field_name, None)

                    # Key-Value verification with the value of its field
                    new_entry = None
                    if value:
                        new_entry = self._dataop.is_in(new_value, value)
                    # new_entry = new_value in value if value else None

                except Exception as e:

                    # Continue and force eligibility
                    if operation == "OR":
                        result_or_list.append(True)
                    elif operation == "AND":
                        result_and_list.append(True)
                    return

                # Logging #############################################################
                log1 = f"####oooo 3. _lookup_eligibility() with UID = {unique_id}:"
                log2 = f"            - Main Loop: key = {key}, value = {value}"
                log3 = f"            - Entered option = VALUE == LIST - Lonely Field has:"
                log4 = f"            - new_field_name = {new_field_name}"
                log5 = f"            - new_value = {new_value}"
                log6 = f"            - new_entry = {new_entry}"
                log7 = f"            - new_current_rules = {new_current_rules}"
                for log in [log1, log2, log3, log4, log5, log6, log7]:
                    log_message.append(log)
                #######################################################################

                if new_entry is not None:

                    # Recursive lookup for dependent fields
                    is_eligible = True
                    if "conditional_fields" in new_current_rules and new_current_rules["conditional_fields"] is not None:

                        # Logging #############################################################
                        log1 = f"####oooooooo 4. _lookup_eligibility() with UID = {unique_id}:"
                        log2 = f"                - Main Loop: key = {key}, value = {value}"
                        log3 = f"                - Entered option = VALUE == LIST - Lonely Field has DEPENDENCIES:"
                        log4 = f"                - Running _lookup_eligibility()."
                        for log in [log1, log2, log3, log4]:
                            log_message.append(log)
                        #######################################################################

                        # Correct format for the next function
                        new_current_rules = new_current_rules["conditional_fields"]

                        # Go back and check eligibility of new conditional field
                        is_eligible = self._check_eligibility(data, row_idx, new_field_name, new_current_rules, log_message)

                        # Logging #############################################################
                        log1 = f"####oooooooo 5. _lookup_eligibility() with UID = {unique_id}:"
                        log2 = f"                - Main Loop: key = {key}, value = {value}"
                        log3 = f"                - Entered option = VALUE == LIST - Lonely Field has DEPENDENCIES:"
                        log4 = f"                - _lookup_eligibility() successfully returned:"
                        log5 = f"                - is_eligible = {is_eligible}"
                        for log in [log1, log2, log3, log4, log5]:
                            log_message.append(log)
                        #######################################################################

                    # Logging #############################################################
                    log1 = f"####oooooo 6. _lookup_eligibility() with UID = {unique_id}:"
                    log2 = f"              - Main Loop: key = {key}, value = {value}"
                    log3 = f"              - Entered option = VALUE == LIST - Lonely Field."

                    # And only append the value to the respective list if it returns True.
                    if is_eligible:
                        if operation == "OR":
                            result_or_list.append(new_entry)
                            log4 = f"              - Lonely Field is eligible {new_entry} was appended to result_or_list."
                        elif operation == "AND":
                            result_and_list.append(new_entry)
                            log4 = f"              - Lonely Field is eligible {new_entry} was appended to result_and_list."
                    else:
                        log4 = f"              - Lonely Field is NOT eligible."

                    log5 = f"              - EXITING current_rules's loop - option LIST."
                    for log in [log1, log2, log3, log4, log5]:
                        log_message.append(log)
                    #######################################################################

                else:

                    if operation == "OR":
                        result_or_list.append(False)
                    elif operation == "AND":
                        result_and_list.append(False)

                    # Logging #############################################################
                    log1 = f"####oooooo 4. _lookup_eligibility() with UID = {unique_id}:"
                    log2 = f"              - Main Loop: key = {key}, value = {value}"
                    log3 = f"              - Entered option = VALUE = LIST - Lonely Field."
                    log4 = f"              - But new_entry is None so False was appended to both results lists."
                    log5 = f"              - EXITING current_rules's loop - option LIST."
                    for log in [log1, log2, log3, log4, log5]:
                        log_message.append(log)
                    #######################################################################

            else:

                # Logging #############################################################
                log1 = f"####oooo E. _lookup_eligibility() with UID = {unique_id}:"
                log2 = f"            - Error at loop: key = {key}, value = {value}"
                log3 = f"            - Exiting: Unknown key,value pair."
                for log in [log1, log2, log3]:
                    log_message.append(log)
                #######################################################################

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    def _collapse_one_hot(self, any_flag=True) -> pd.DataFrame:
        """
        Collapse one-hot encoded columns into a single categorical column per cluster.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the one-hot encoded columns.
        rules (dict): The YAML dictionary containing column metadata.

        Returns:
        pd.DataFrame: The transformed DataFrame with collapsed one-hot fields.
        """

        # Get all fields from data frame
        all_fields = self.data.columns.to_list()

        # Create a copy of the DataFrame to modify
        collapsed_data = self.data.copy(deep=True)

        # Create new dictionary
        new_rules = copy.deepcopy(self.yaml_rules)
        fields_to_remove = []  # Collect fields to remove

        # Identify all one-hot clusters
        one_hot_clusters = {}
        for field, rule in self.yaml_rules.items():

            # Check if field in dataframe
            if field not in all_fields:
                continue

            # If field is one-hot-cluster
            if new_rules[field]["one_hot_cluster"] is not None:

                # Marks one-hot-cluster field for deletion - ALL - even if "not eligible"
                fields_to_remove.append(field) # Mark for deletion

                # Name of one-hot-cluster
                cluster_name = new_rules[field]["one_hot_cluster"]["name"]
                
                # If it is the first time of this one-hot-cluster
                if cluster_name not in one_hot_clusters:
                    one_hot_clusters[cluster_name] = []

                # Append eligible field to one_hot_clusters
                one_hot_clusters[cluster_name].append(field)

                # If satisfies "eligible," append the field and track attributes
                if new_rules[field]["one_hot_cluster"]["eligible"]:

                    # Initialize cluster attributes if not already set
                    if cluster_name not in new_rules:
                        new_rules[cluster_name] = {}

                    # Merge attributes (preserve first seen value)
                    for attr, value in new_rules[field].items():
                        if attr not in new_rules[cluster_name]: # Only add if it doesn’t exist
                            new_rules[cluster_name][attr] = value

        # Create new dataframe
        for cluster_name, fields in one_hot_clusters.items():

            new_column_name = cluster_name  # The name of the new column
    
            if any_flag: # Create data for 'missing data'

                # Create a new column where each row contains True if at least one non-null values from the cluster is True
                collapsed_data[new_column_name] = collapsed_data[fields].apply(
                    lambda row: any([val for field, val in row.items() if new_rules[field]["one_hot_cluster"]["eligible"]]),
                    axis=1
                )

            else:

                # Create a new column where each row contains a CIGAR string
                collapsed_data[new_column_name] = collapsed_data[fields].apply(
                    lambda row: "".join("1" if val else "0" for field, val in row.items()),
                    axis=1
                )

            # Drop original one-hot columns
            collapsed_data.drop(columns=fields, inplace=True)

        # Remove after iteration
        for field in fields_to_remove:
            new_rules.pop(field, None)

        return collapsed_data, new_rules

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    # UNIQUENESS MAIN:

    def calculate_uniqueness(self) -> Dict[str, np.float32]:
        """
        Calculate the uniqueness score for the dataset.

        Uniqueness is determined by identifying and counting duplicate rows based on specified rules.

        Returns
        -------
        NDArray[np.float32]
            Array with [duplicate_count, total_duplicate_count, uniqueness_score].

        Raises
        ------
        DuplicateDataError
            If duplicate data is detected.
        CompletenessError
            For generic completeness issues.
        """

        try:

            results = self._calculate_duplicate(self.data, create_duplicate_structure=True, create_duplicate_report=True)

            # Ensure the function returned exactly 3 values
            if not isinstance(results, (list, tuple)) or len(results) != 3:
                raise CompletenessError("Function `_calculate_duplicate` did not return exactly 3 values.")

            # Create dictionary
            name_vector = ["Total Duplicate Values: ", "Total Table Values: ", "Uniqueness: "]
            result_vector = [np.float32(e) if isinstance(e, float) else np.int16(e) for e in results]
            full_result = {name: result for name, result in zip(name_vector, result_vector)}

            # Convert result to NumPy array
            return full_result

        except CompletenessError:  
            # Propagates
            raise
        except DuplicateDataError:
            raise
        except Exception as e:  
            # Any other error, wrap in CompletenessError
            raise CompletenessError("Error calculating uniqueness") from e

    def calculate_bagging_uniqueness(self) -> Dict[str, np.float32]:
        """
        Calculate uniqueness using bagging.

        Bagging involves repeatedly sampling rows and columns with replacement to estimate variance in uniqueness.

        Returns
        -------
        NDArray[np.float32]
            Array with concatenated results for row-based and column-based uniqueness.

        Raises
        ------
        DuplicateDataError
            If an error occurs during uniqueness calculation.
        """

        n_rows, n_cols = self.data.shape
        bagging_size_row = n_rows
        bagging_size_col = n_cols
        bagging_times = self.global_yaml.yaml_dict['completeness_metric']['bagging_times']
        duplication_tolerance = self.global_yaml.yaml_dict['completeness_metric']['duplication_tolerance']

        row_result_vector = []
        col_result_vector = []
        random_row_result_vector = []
        random_col_result_vector = []

        # Create mock random completeness dataset
        try:
            resvec = self._calculate_duplicate(self.data)
            # Ensure the function returned exactly 3 values
            if not isinstance(resvec, (list, tuple)) or len(resvec) != 3:
                raise CompletenessError("Function `_calculate_duplicate` did not return exactly 3 values.")
            random_data = self._generate_random_uniqueness(self.data, resvec[-1], duplication_tolerance)
        except CompletenessError:  
            # Propagation
            raise
        except DuplicateDataError as e:
            print(e)
            raise DuplicateDataError(e)
        except Exception as e:  
            # Any other error, wrap in CompletenessError
            raise CompletenessError("Error calculating completeness") from e

        # Generate bagging results
        try:

            for i in range(bagging_times):

                sampled_rows = self.data.sample(n=bagging_size_row, replace=True)
                sampled_cols = self.data.sample(n=bagging_size_col, axis=1, replace=True)
                random_rows = random_data.sample(n=bagging_size_row, replace=True)
                random_cols = random_data.sample(n=bagging_size_col, axis=1, replace=True)

                row_result = self._calculate_duplicate(sampled_rows)
                # Ensure the function returned exactly 3 values
                if not isinstance(row_result, (list, tuple)) or len(row_result) != 3:
                    raise CompletenessError("Function `_calculate_duplicate` did not return exactly 3 values.")

                row_result_vector.append(row_result[-1])

                col_result = self._calculate_duplicate(sampled_cols)
                # Ensure the function returned exactly 3 values
                if not isinstance(col_result_vector, (list, tuple)) or len(col_result_vector) != 3:
                    raise CompletenessError("Function `_calculate_duplicate` did not return exactly 3 values.")

                col_result_vector.append(col_result[-1])

                random_row_result = self._calculate_duplicate(random_rows)
                # Ensure the function returned exactly 3 values
                if not isinstance(random_row_result, (list, tuple)) or len(random_row_result) != 3:
                    raise CompletenessError("Function `_calculate_duplicate` did not return exactly 3 values.")

                random_row_result_vector.append(random_row_result[-1])

                random_col_result = self._calculate_duplicate(random_cols)
                # Ensure the function returned exactly 3 values
                if not isinstance(random_col_result, (list, tuple)) or len(random_col_result) != 3:
                    raise CompletenessError("Function `_calculate_duplicate` did not return exactly 3 values.")

                random_col_result_vector.append(random_col_result[-1])

            # Row and column numpy arrays
            row_result_vector = np.array(row_result_vector, dtype=np.float32)
            col_result_vector = np.array(col_result_vector, dtype=np.float32)
            random_row_result_vector = np.array(random_row_result_vector, dtype=np.float32)
            random_col_result_vector = np.array(random_col_result_vector, dtype=np.float32)

            # Mean and std vector
            result_vector = np.array([ np.mean(row_result_vector),
                                       np.std(row_result_vector, ddof=1),
                                       np.mean(col_result_vector),
                                       np.std(col_result_vector, ddof=1),
                                       np.mean(random_row_result_vector),
                                       np.std(random_row_result_vector, ddof=1),
                                       np.mean(random_col_result_vector),
                                       np.std(random_col_result_vector, ddof=1),
                                       ], dtype=np.float32)

            # Name vector
            name_vector = ["Row Completeness Mean",
                           "Row Completeness Std",
                           "Column Completeness Mean",
                           "Column Completeness Std",
                           "Random Row Completeness Mean",
                           "Random Row Completeness Std",
                           "Random Column Completeness Mean",
                           "Random Column Completeness Std",
                           ]

            # Create dictionary with zip
            full_result = {name: result for name, result in zip(name_vector, result_vector)}

            return full_result

        except CompletenessError:  
            # Propagation
            raise
        except DuplicateDataError as e:
            print(e)
            raise DuplicateDataError(e)
        except Exception as e:  
            # Any other error, wrap in CompletenessError
            raise CompletenessError("Error calculating completeness") from e

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    # DUPLICATION:

    def _calculate_duplicate(self,
                             data : pd.DataFrame,
                             create_duplicate_structure : bool = False,
                             create_duplicate_report : bool = False) -> tuple:
        """
        Detect and count duplicate rows based on specified rules.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to check for duplicates.
        rules : Dict[Any, Any]
            Dictionary with rules and parameters to check for duplicates.
        create_duplicate_structure : bool
            If True, stores the duplicate rows for later inspection.
        create_duplicate_structure : bool
            If True, stores a detailed report with missing data.

        Returns
        -------
        tuple
            A tuple with (duplicate_count, total_duplicate_count, uniqueness_score).

        Raises
        ------
        DuplicateDataError
            If duplicate data is detected.
        """

        # USE: self._dataop.compare(value, value)
        # USE: self._dataop.is_in(value, value)
        # USE: self._dataop.is_not_in(value, value)

        # Function's unique ID
        unique_id = str(np.random.randint(0, 999)).zfill(3)

        # Initialize log messages
        log_message = []

        # Logging #############################################################
        # log1 = f"1. ENTERING _calculate_duplicate() with UID = {unique_id}:"
        # log2 = f"            - XXXXXXX = {XXXXXXXXX}"
        # log3 = f"            - XXXXXXX = {XXXXXXXXX}"
        # for log in [log1, log2, log3]:
        #     log_message.append(log)
        #######################################################################

        # Initialize score count
        duplicate_count = 0
        total_duplicate_count = 0
        missing_weight = 1

        # Initialize helper structures
        previous_rows = OrderedDict()

        # Initialize duplication tolerance
        duplication_tolerance = self.global_yaml.yaml_dict['completeness_metric']['duplication_tolerance']

        # Iterate through each row to detect duplicates
        for current_row_idx, current_row in self.data.iterrows():

            # Converting pd.Series to a dictionary
            current_row_dict = current_row.to_dict()

            # Base case - Previous rows is empty
            if not previous_rows:
                previous_rows[current_row_idx] = current_row_dict
                continue

            # Get list of fields 
            list_of_fields = current_row_dict.keys()

            # Compare current_row with previously processed rows
            for previous_row_idx, previous_row_dict in previous_rows.items():

                try:
                    similarity = self._calculate_duplicate_instance(data,
                                                                    previous_row_idx,
                                                                    previous_row_dict,
                                                                    current_row_idx,
                                                                    current_row_dict,
                                                                    duplication_tolerance,
                                                                    list_of_fields,
                                                                    log_message)
                except Exception as e:
                    raise DuplicateDataError((f"Failed to calculate similarity for rows "
                                              f"{previous_row_idx} and {current_row_idx}"))

                if similarity >= duplication_tolerance:

                    duplicate_count += missing_weight

                    # Missing Structure
                    if create_duplicate_structure:
                        self._missing_locations.append((previous_row_idx,
                                                        current_row_idx, 
                                                        similarity,
                                                        missing_weight))

                    # Missing Report
                    if create_duplicate_report:
                        if previous_row_idx not in self._missing_report:
                            self._missing_report[previous_row_idx] = OrderedDict()
                        if current_row_idx not in self._missing_report[previous_row_idx]:
                            self._missing_report[previous_row_idx][current_row_idx] = OrderedDict()
                        all_row_1 = ", ".join([str(e) for e in previous_row_dict.values()])
                        all_row_2 = ", ".join([str(e) for e in current_row_dict.values()])
                        result_list = [similarity, duplication_tolerance, all_row_1, all_row_2, 
                                       log_message]
                        self._missing_report[previous_row_idx][current_row_idx] = result_list

                # Updating total count         
                total_duplicate_count += missing_weight

            # Append current objects to previous list
            previous_rows[current_row_idx] = current_row_dict

        uniqueness_score = 1 - (duplicate_count / total_duplicate_count) if total_duplicate_count else 1.0

        return duplicate_count, total_duplicate_count, uniqueness_score

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    # DUPLICATION HELPERS:

    def _calculate_duplicate_instance(self,
                                      data : pd.DataFrame,
                                      previous_row_idx : int,
                                      previous_row_dict : Dict[int, Any],
                                      current_row_idx : int,
                                      current_row_dict : Dict[int, Any],
                                      duplication_tolerance : float,
                                      list_of_fields : List,
                                      log_message : List) -> tuple:
        """
        Detect and count duplicate rows based on specified rules.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to check for duplicates.
        rules : Dict[Any, Any]
            Dictionary with rules and parameters to check for duplicates.
        create_duplicate_structure : bool
            If True, stores the duplicate rows for later inspection.
        create_duplicate_structure : bool
            If True, stores a detailed report with missing data.

        Returns
        -------
        tuple
            A tuple with (duplicate_count, total_duplicate_count, uniqueness_score).

        Raises
        ------
        DuplicateDataError
            If duplicate data is detected.
        """

        # USE: self._dataop.compare(value, value)
        # USE: self._dataop.is_in(value, value)
        # USE: self._dataop.is_not_in(value, value)
        # current_rules = self.yaml_rules[field]

        # Function's unique ID
        unique_id = str(np.random.randint(0, 999)).zfill(3)

        # Logging #############################################################
        # log1 = f"1. ENTERING _calculate_duplicate_instance() with UID = {unique_id}:"
        # log2 = f"            - XXXXXXX = {XXXXXXXXX}"
        # log3 = f"            - XXXXXXX = {XXXXXXXXX}"
        # for log in [log1, log2, log3]:
        #     log_message.append(log)
        #######################################################################

        # Initialize cumulative similarity and total weight trackers
        matching_weight = 0.0  # Sum of field weights where values match
        total_weight = 0.0     # Sum of all field weights
        
        # Iterate on fields (columns)
        for field in list_of_fields:

            # Current rules
            current_rules = self.yaml_rules[field]
            weight = current_rules["duplication_weight"]

            # Check match
            try:
                match = self._dataop.compare(previous_row_dict[field], current_row_dict[field])
            except Exception as e:  
                # Any other error, wrap in CompletenessError
                raise DuplicateDataError("In function '_calculate_duplicate_instance'.") from e

            if not match:
                continue
                total_weight += weight

            # Check if field is forbidden
            if current_rules["duplication_forbidden"]:
                return 1.0

            # Check dependent fields
            is_eligible_1 = True
            is_eligible_2 = True
            if "conditional_fields" in current_rules and current_rules["conditional_fields"] is not None:
                current_cond = current_rules["conditional_fields"]
                is_eligible_1 = self._check_eligibility(data, previous_row_idx, field, current_cond, log_message)
                is_eligible_2 = self._check_eligibility(data, current_row_idx, field, current_cond, log_message)             

            # Final check
            if is_eligible_1 is not None and is_eligible_2 is not None and is_eligible_1 and is_eligible_2:
                matching_weight += weight
                total_weight += weight

        # Calculating similarity
        similarity = matching_weight / total_weight if total_weight else 0.0
        return similarity

    """
    def _calculate_similarity(self,
                              data: pd.DataFrame,
                              row_1: pd.Series,
                              row_2: pd.Series,
                              rules: Dict,
                              field_weights: Dict[str, float],
                              field_onehot: Dict[str, str]) -> float:

        Compute the similarity score between two rows in a Pandas DataFrame using field-specific weights.
        
        The similarity score ranges from **0.0 to 1.0**, where:
            - **1.0** indicates a perfect match across all fields.
            - **0.0** indicates no matching fields.
        
        Parameters
        ----------
        row_1 : pd.Series
            The first row (observation) from the DataFrame.
            
        row_2 : pd.Series
            The second row (observation) from the DataFrame.
            
        field_weights : Dict{str, float}
            A dictionary of numerical weights, where each value corresponds to the **importance** of a field.
            This dictionary **must have the same length** as the number of columns in the original DataFrame.

        Returns
        -------
        float
            A similarity score **between 0.0 and 1.0**, where higher values indicate stronger similarity.

        Notes
        -----
        - The function **strictly** compares values using the brute-force method:
          `if row_1[field] == row_2[field] then add weight, otherwise do not add`.
        - **No shortcuts** (e.g., NumPy vectorization) are used for comparisons.
        - The function ensures that each field contributes its weight to a **running total**,
          preventing any precomputed sums.
          
        Example
        -------
        >>> df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y'], 'C': [3.5, 3.5]})
        >>> row_1 = df.iloc[0]
        >>> row_2 = df.iloc[1]
        >>> field_weights = [0.5, 0.3, 0.2]
        >>> similarity = _calculate_similarity(None, row_1, row_2, field_weights)
        >>> print(similarity)
        >>> 0.2

        # Initialize cumulative similarity and total weight trackers
        matching_weight = 0.0  # Sum of field weights where values match
        total_weight = 0.0     # Sum of all field weights

        # Extract column names (assuming both rows are from the same DataFrame)
        columns = row_1.index  # This retrieves the column labels from the Pandas Series
        all_fields = field_weights.keys() # Retrieves all fields from the field_weights

        # Ensure the number of weights matches the number of fields
        assert len(field_weights) == len(columns), (
            f"Mismatch between field count ({len(columns)}) and field_weights size ({len(field_weights)})!"
        ) # Future = Throw Exception

        # Flag one-hot to pass only one time per one-hot
        flag_one_hot = [] # TODOOOOOOOOOOOOOOOOOOOOOOOOOOO# TODOOOOOOOOOOOOOOOOOOOOOOOOOOO

        # Iterate through each field in the row, comparing values manually
        for idx, field in enumerate(columns):

            # Implmenentation of YAMLFieldNotFound 
            try:
                # In case the field of the table was not defined in the YAML
                # In future, could put here all "default" fields. And have a
                # Separate for field does not exist in YAML for real.
                if field not in all_fields: # TODOOOOOOOOOOOOOOOOOOOOOOOOOOO# TODOOOOOOOOOOOOOOOOOOOOOOOOOOO
                    raise YAMLFieldNotFound(
                        message=f"Field '{field}' not found in YAML configuration.",
                        file_path=".this.rules.table.yaml",
                        rule_name="YAMLFieldExistence",
                        conflicting_table_field=field,
                        severity='warning'  # For now, it is a warning
                    )
                else:
                    field_weight = field_weights[field]  # Retrieve the weight for this field
            except YAMLFieldNotFound as e:
                if e.severity == 'error':
                    raise  # Stop execution if it's an actual error
                else:
                    print(f"Warning: {e.message}")  # Just log a warning for now

            # Explicitly check if values are equal
            if row_1[field] == row_2[field]:

                # Check missing values # TODOOOOOOOOOOOOOOOOOOOOOOOOOOO
                missing_1 = self.missing_values[idx][field]
                missing_2 = self.missing_values[previous_idx][field]
                missing_3 = pd.isna(data.at[idx, field])
                missing_4 = pd.isna(data.at[previous_idx, field])
                if missing_1 or missing_2 or missing_3 or missing_4:
                    continue_signal = True
                    continue                   

                # Conditional field # TODOOOOOOOOOOOOOOOOOOOOOOOOOOO
                if (
                    self.yaml_rules['conditional_fields'] is not None
                    and len(self.yaml_rules['conditional_fields']) >= 1
                ):

                    # TODO
                    pass

                # Add weight if values match # TODOOOOOOOOOOOOOOOOOOOOOOOOOOO
                matching_weight += field_weight  
                total_weight += field_weight

            else:

                # Ensure weight is always included in total weight # TODOOOOOOOOOOOOOOOOOOOOOOOOOOO
                total_weight += field_weight
        
        # Calculate final similarity score (avoid division by zero)
        similarity_score = matching_weight / total_weight if total_weight > 0 else 0.0

        return similarity_score
    """

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    # RANDOM GENERATORS:

    def _generate_random_completeness(reference_df: pd.DataFrame, completeness: float) -> pd.DataFrame:
        """
        Generate a mock numeric dataset with specified completeness, based on the shape
        and column names of a reference DataFrame.

        Parameters
        ----------
        reference_df : pd.DataFrame
            The reference DataFrame used to extract shape and column names.
        completeness : float
            Desired completeness (between 0 and 1), where:
            completeness = 1 - (number_of_missing / total_cells)

        Returns
        -------
        pd.DataFrame
            A mock DataFrame with the same shape and columns as `reference_df`, filled with random integers [0-9],
            and missing values (np.nan) placed at random positions to match the target completeness.
        """
        assert 0 <= completeness <= 1, "completeness must be between 0 and 1"

        n_rows, n_cols = reference_df.shape
        total_cells = n_rows * n_cols
        n_missing = int((1 - completeness) * total_cells)

        # Generate random integers between 0 and 9
        mock_data = np.random.randint(0, 10, size=(n_rows, n_cols)).astype(float)

        # Create random positions to set as NaN
        all_indices = [(i, j) for i in range(n_rows) for j in range(n_cols)]
        missing_indices = np.random.choice(len(all_indices), size=n_missing, replace=False)
        for idx in missing_indices:
            i, j = all_indices[idx]
            mock_data[i, j] = pd.NA

        # Convert to DataFrame with original column names
        mock_df = pd.DataFrame(mock_data, columns=reference_df.columns)

        return mock_df

    def _generate_random_uniqueness(reference_df: pd.DataFrame,
                                    uniqueness: float,
                                    duplicate_threshold: float
                                    ) -> pd.DataFrame:
        """
        Generate a mock numeric dataset with specified uniqueness, based on the shape
        and column names of a reference DataFrame.

        Parameters
        ----------
        reference_df : pd.DataFrame
            The reference DataFrame used to extract shape and column names.
        uniqueness : float
            Desired uniqueness (between 0 and 1), where:
            uniqueness = 1 - (number_of_duplicate_row_pairs / total_possible_row_pairs)
        duplicate_threshold : float
            Proportion of identical columns required to consider a pair of rows as duplicates.

        Returns
        -------
        pd.DataFrame
            A mock DataFrame with the same shape and columns as `reference_df`, filled with random integers [0-9],
            and rows partially duplicated in a completely random way to match the target uniqueness.
        """

        assert 0 <= uniqueness <= 1, "uniqueness must be between 0 and 1"
        assert 0 <= duplicate_threshold <= 1, "duplicate_threshold must be between 0 and 1"

        n_rows, n_cols = reference_df.shape
        total_pairs = max((n_rows * (n_rows - 1)) // 2, 0)
        n_duplicate_pairs = int((1 - uniqueness) * total_pairs)

        # Step 1: Generate base random data
        mock_data = np.random.randint(0, 10, size=(n_rows, n_cols))

        # Step 2: Select random row pairs to apply partial duplication
        all_possible_pairs = list(combinations(range(n_rows), 2))
        chosen_pairs = sample(all_possible_pairs, min(n_duplicate_pairs, len(all_possible_pairs)))

        for i, j in chosen_pairs:
            n_equal_fields = int(np.ceil(duplicate_threshold * n_cols))
            equal_indices = sample(range(n_cols), n_equal_fields)
            for col_idx in equal_indices:
                mock_data[j, col_idx] = mock_data[i, col_idx]

        # Step 3: Convert to DataFrame
        mock_df = pd.DataFrame(mock_data, columns=reference_df.columns)

        return mock_df

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    # OUTPUT FUNCTIONS:

    def case(self) -> Generator:
        """
        Generator yielding missing and duplicate cases.

        Yields
        ------
        tuple
            Yields ('Missing', location, None) or ('Duplicate', row_1, row_2)
        """
        if self._missing_locations and self._missing_locations is not None:
            for miss in self._missing_locations:
                yield ('Missing', *miss)
        if self._duplicate_rows and self._duplicate_rows is not None:
            for dupl in self._duplicate_rows:
                yield ('Duplicate', *dupl)

    def report(self, output_path: str, type: str="completeness") -> None:
        """
        Generate a report of missing and duplicate cases.

        Parameters
        ----------
        output_path : str
            File path to save the report.
        """
        if(type == "completeness"):
            return self.report_completeness(output_path)
        elif(type == "uniqueness"):
            return self.report_uniqueness(output_path)

    def report_completeness(self, output_path: str) -> None:
        """
        Generate a report of missing and duplicate cases.

        Parameters
        ----------
        output_path : str
            File path to save the report.
        """

        hash_line = f"#"*124
        dash_line = f"-"*124

        with open(output_path, 'w') as file:
            for row in self._missing_report:
                for field in self._missing_report[row]:

                    to_write = [
                        hash_line+f"\n",
                        f"Row: {self._missing_report[row][field][0]}\t\t\t\t",
                        f"Field: {self._missing_report[row][field][1]}\t\t\t",
                        f"Value: {self._missing_report[row][field][2]}\n",
                        f"Missing: {self._missing_report[row][field][3]}\t\t",
                        f"Status: {self._missing_report[row][field][4]}\t\t\t\t",
                        f"Weight: {self._missing_report[row][field][5]}\n",
                        f"Field Rules:\n"]

                    self.global_yaml.print_tree(self._missing_report[row][field][6], to_write, "", False)
                    to_write[-1] = "".join(to_write[-1][:-1])

                    for value in [dash_line, f"LOG:", dash_line] + self._missing_report[row][field][7]:
                        to_write.append(f"\n"+value)

                    to_write.append(f"\n"+dash_line+f"\n\n")

                    file.write("".join(to_write))

    def report_uniqueness(self, output_path: str) -> None:
        """
        Generate a report of missing and duplicate cases.

        Parameters
        ----------
        output_path : str
            File path to save the report.
        """

        hash_line = f"#"*124
        dash_line = f"-"*124

        with open(output_path, 'w') as file:
            for row1 in self._duplicate_report:
                for row2 in self._duplicate_report[row]:

                    to_write = [
                        hash_line+f"\n",
                        f"Row_1: {row1}\t\t",
                        f"Row_2: {row2}\t\t",
                        f"Similarity: {self._duplicate_report[row1][row2][0]}\t\t",
                        f"Tolerance: {self._duplicate_report[row1][row2][1]}\n",
                        f"Full_Row_1: {self._duplicate_report[row1][row2][2]}\n",
                        f"Full_Row_2: {self._duplicate_report[row1][row2][3]}\n",
                    ]

                    for value in [dash_line, f"LOG:", dash_line] + self._duplicate_report[row1][row2][4]:
                        to_write.append(f"\n"+value)

                    to_write.append(f"\n"+dash_line+f"\n\n")

                    file.write("".join(to_write))


    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    # DESCRIPTION FUNCTIONS:

    def description(self) -> str:
        """
        Provides a textual description of the completeness metric.

        Returns
        -------
        str
            Description in Portuguese.
        """
        return (
            "Um conjunto de dados completo significa que todos os campos que "
            "podem ter algum valor associado têm tais valores associados preenchidos "
            "e que não existem registros duplicados. Isso é essencial para análises "
            "precisas, pois tanto a falta de dados quanto duplicações dos dados pode "
            "comprometer a interpretação e os resultados de uma pesquisa."
        )

    def __repr__(self) -> str:
        """
        Machine-readable representation of the object.

        Returns
        -------
        str
            Machine-readable class description.
        """
        return (
            f"CompletenessMetric(data={self.data.shape}, missing_values={self.missing_values!r}, "
            f"global_yaml={self.global_yaml.input_path}, completeness_yaml={self.completeness_yaml.input_path})"
        )

    def __str__(self) -> str:
        """
        Human-readable string representation.

        Returns
        -------
        str
            Human-readable class description.
        """

        return (
            f"CompletenessMetric with {len(self.data)} rows and {len(self.data.columns)} columns; "
            f"Missing: {len(self._missing_locations)} cells; Duplicates: {len(self._duplicate_rows)} rows."
        )

class ConformanceMetric(QualityMetric):
    """
    Measures the conformance of the dataset to predefined rules or standards.

    Conformance is defined as the proportion of data entries that adhere
    to a set of rules, such as valid value ranges, data types, or domain-specific
    constraints.
    """

    def __init__(self, data: Any, rules: Dict[str, Callable[[Any], bool]]):
        """
        Initializes the ConformanceMetric with data and validation rules.

        Parameters:
        ----------
        data : Any
            The dataset on which the conformance metric will be calculated.
        rules : Dict[str, Callable[[Any], bool]]
            A dictionary where keys are field names and values are callable
            functions that define validation rules for the corresponding field.
            Each function should take a value and return True if the value conforms.
        """
        super().__init__(data)
        self.rules = rules

    def calculate(self) -> np.float64:
        """
        Calculates the conformance score for the dataset.

        Conformance is computed as:
        (Number of valid entries) / (Total number of entries).

        Returns:
        -------
        np.float64
            The conformance score, where 1.0 indicates full conformance.
        """
        # Implementation logic goes here
        pass

    def case(self) -> Generator[Any, None, None]:
        """
        Yields individual cases (e.g., rows, columns, or fields) that violate
        the conformance rules.

        Yields:
        ------
        Any
            The specific cases that contribute to a lower conformance score,
            such as field names or row identifiers.
        """
        # Implementation logic goes here
        pass

    def report(self) -> List[Any]:
        """
        Generates a comprehensive report of all cases that violate the conformance rules.

        Returns:
        -------
        List[Any]
            A list of cases (e.g., row/field identifiers) that negatively impact
            the conformance score.
        """
        # Implementation logic goes here
        pass

    def description(self) -> str:
        """
        Provides a description of the ConformanceMetric.

        Returns:
        -------
        str
            A string describing the metric and its methodology.
        """
        return (
            "Dados conformes seguem regras predefinidas, como formatos de campos "
            "ou regras de integridade referencial. Por exemplo, se a regra para "
            "exibir/colocar datas seja no formato 'DD/MM/AAAA'. Não se podem "
            "colocar: 'MM/DD/AAAA', 'DD/MM/AA' ou 'DD-MM-AAAA'."
        )

    def __repr__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A string representation for debugging and development purposes.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> repr(placeholder)
        "Placeholder(data=<placeholder>, missing_values=<placeholder>"
        """
        return (
            f"Placeholder(data={self.data!r}, missing_values={self.missing_values!r})"
        )

    def __str__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A simplified representation of the object's core content.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> str(placeholder)
        "Placeholder with data: <placeholder>, missing_values: <placeholder>"
        """
        return (
            f"Placeholder with data: '{self.value}', "
            f"missing_values: {self.missing_values}"
        )

class ConsistencyMetric(QualityMetric):
    """
    Measures the consistency of the dataset by evaluating internal coherence 
    across fields or within a field over multiple entries.

    Consistency is defined as the degree to which data does not contain
    contradictory or conflicting information. It checks logical relationships
    within and across fields, such as dependencies, invariants, or correlations.
    """

    def __init__(self, data: Any, rules: List[Tuple[str, Callable[[Any], bool]]]):
        """
        Initializes the ConsistencyMetric with data and validation rules.

        Parameters:
        ----------
        data : Any
            The dataset on which the consistency metric will be calculated.
        rules : List[Tuple[str, Callable[[Any], bool]]]
            A list of tuples, where each tuple consists of:
              - A description of the rule (as a string).
              - A callable function that evaluates consistency logic. 
                The function should take a dataset or subset and return True
                if the data is consistent.
        """
        super().__init__(data)
        self.rules = rules

    def calculate(self) -> np.float64:
        """
        Calculates the consistency score for the dataset.

        Consistency is computed as:
        (Number of consistent cases) / (Total number of cases checked).

        Returns:
        -------
        np.float64
            The consistency score, where 1.0 indicates full consistency.
        """
        # Implementation logic goes here
        pass

    def case(self) -> Generator[Any, None, None]:
        """
        Yields individual cases (e.g., rows, columns, or relationships) 
        that violate consistency rules.

        Yields:
        ------
        Any
            Specific cases or identifiers contributing to a lower consistency score.
        """
        # Implementation logic goes here
        pass

    def report(self) -> List[Any]:
        """
        Generates a comprehensive report of all cases that violate consistency rules.

        Returns:
        -------
        List[Any]
            A list of cases or relationships that negatively impact the consistency score.
        """
        # Implementation logic goes here
        pass

    def description(self) -> str:
        """
        Provides a description of the ConsistencyMetric.

        Returns:
        -------
        str
            A string describing the metric and its methodology.
        """
        return (
            "A consistência garante que as mesmas regras e formatos sejam aplicados "
            "em todas as instâncias de dados. Por exemplo, se uma variável é registrada "
            "como 'Masculino' e 'Feminino' em um campo, ela não deve aparecer como 'M' e "
            "'F' em outro campo, ainda que diferente. Implementar a consistência pode "
            "envolver padronização de formatos e aplicação de regras de validação que "
            "verifiquem uniformidade entre os diferentes campos possíveis."
        )

    def __repr__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A string representation for debugging and development purposes.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> repr(placeholder)
        "Placeholder(data=<placeholder>, missing_values=<placeholder>"
        """
        return (
            f"Placeholder(data={self.data!r}, missing_values={self.missing_values!r})"
        )

    def __str__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A simplified representation of the object's core content.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> str(placeholder)
        "Placeholder with data: <placeholder>, missing_values: <placeholder>"
        """
        return (
            f"Placeholder with data: '{self.value}', "
            f"missing_values: {self.missing_values}"
        )

class AgreementMetric(QualityMetric):
    """
    Measures the agreement between multiple sources, raters, or fields 
    within a dataset.

    Agreement is defined as the degree to which values from different sources 
    or raters align or agree with one another. This can be evaluated using 
    metrics like Cohen's kappa, intraclass correlation coefficient (ICC), or 
    simple percentage agreement, depending on the use case.
    """

    def __init__(self, data: Any, comparisons: List[Dict[str, Callable[[Any], bool]]]):
        """
        Initializes the AgreementMetric with data and comparison logic.

        Parameters:
        ----------
        data : Any
            The dataset on which the agreement metric will be calculated.
        comparisons : List[Dict[str, Callable[[Any], bool]]]
            A list of dictionaries where:
              - Keys describe the comparison (e.g., "Rater1 vs Rater2").
              - Values are callable functions that define how agreement 
                is calculated for the comparison. Each function should take
                relevant data and return True if the values agree.
        """
        super().__init__(data)
        self.comparisons = comparisons

    def calculate(self) -> np.float64:
        """
        Calculates the agreement score for the dataset.

        Agreement is computed as:
        (Number of agreeing cases) / (Total number of comparisons).

        Returns:
        -------
        np.float64
            The agreement score, where 1.0 indicates perfect agreement.
        """
        # Implementation logic goes here
        pass

    def case(self) -> Generator[Any, None, None]:
        """
        Yields individual cases (e.g., rows, columns, or rater pairs) 
        where agreement is violated.

        Yields:
        ------
        Any
            Specific cases or identifiers that contribute to a lower agreement score.
        """
        # Implementation logic goes here
        pass

    def report(self) -> List[Any]:
        """
        Generates a comprehensive report of all cases with disagreement.

        Returns:
        -------
        List[Any]
            A list of cases or comparisons that negatively impact the agreement score.
        """
        # Implementation logic goes here
        pass

    def description(self) -> str:
        """
        Provides a description of the AgreementMetric.

        Returns:
        -------
        str
            A string describing the metric and its methodology.
        """
        return (
            "AgreementMetric: Evaluates the degree of alignment between multiple "
            "sources, raters, or fields in a dataset, using measures such as "
            "Cohen's kappa or percentage agreement."
        )

    def __repr__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A string representation for debugging and development purposes.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> repr(placeholder)
        "Placeholder(data=<placeholder>, missing_values=<placeholder>"
        """
        return (
            f"Placeholder(data={self.data!r}, missing_values={self.missing_values!r})"
        )

    def __str__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A simplified representation of the object's core content.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> str(placeholder)
        "Placeholder with data: <placeholder>, missing_values: <placeholder>"
        """
        return (
            f"Placeholder with data: '{self.value}', "
            f"missing_values: {self.missing_values}"
        )

class RelevanceMetric(QualityMetric):
    """
    Measures the relevance of data with respect to specific use cases or objectives.

    Relevance is defined as the degree to which data contributes to answering
    specific questions or fulfilling its intended purpose. It is often assessed 
    based on context, domain-specific criteria, or predefined importance weights.
    """

    def __init__(self, data: Any, criteria: List[Callable[[Any], bool]]):
        """
        Initializes the RelevanceMetric with data and evaluation criteria.

        Parameters:
        ----------
        data : Any
            The dataset on which the relevance metric will be calculated.
        criteria : List[Callable[[Any], bool]]
            A list of callable functions, where each function represents a criterion 
            for evaluating the relevance of data. Functions should take a data point
            or subset as input and return True if relevant, False otherwise.
        """
        super().__init__(data)
        self.criteria = criteria

    def calculate(self) -> np.float64:
        """
        Calculates the relevance score for the dataset.

        Relevance is computed as:
        (Number of relevant cases) / (Total number of cases evaluated).

        Returns:
        -------
        np.float64
            The relevance score, where 1.0 indicates full relevance.
        """
        # Implementation logic goes here
        pass

    def case(self) -> Generator[Any, None, None]:
        """
        Yields individual cases (e.g., rows or fields) that fail to meet relevance criteria.

        Yields:
        ------
        Any
            Specific cases or identifiers that contribute to a lower relevance score.
        """
        # Implementation logic goes here
        pass

    def report(self) -> List[Any]:
        """
        Generates a comprehensive report of all cases that fail relevance criteria.

        Returns:
        -------
        List[Any]
            A list of cases that negatively impact the relevance score.
        """
        # Implementation logic goes here
        pass

    def description(self) -> str:
        """
        Provides a description of the RelevanceMetric.

        Returns:
        -------
        str
            A string describing the metric and its methodology.
        """
        return (
            "RelevanceMetric: Evaluates the alignment of data with specific use cases "
            "or objectives, assessing its contribution to meaningful insights or decisions."
        )

    def __repr__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A string representation for debugging and development purposes.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> repr(placeholder)
        "Placeholder(data=<placeholder>, missing_values=<placeholder>"
        """
        return (
            f"Placeholder(data={self.data!r}, missing_values={self.missing_values!r})"
        )

    def __str__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A simplified representation of the object's core content.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> str(placeholder)
        "Placeholder with data: <placeholder>, missing_values: <placeholder>"
        """
        return (
            f"Placeholder with data: '{self.value}', "
            f"missing_values: {self.missing_values}"
        )

class RepresentativenessMetric(QualityMetric):
    """
    Measures the representativeness of the dataset in relation to a target population 
    or domain.

    Representativeness is defined as the extent to which the dataset reflects 
    the characteristics, proportions, or diversity of the target population. 
    It often involves comparisons to external benchmarks or expected distributions.
    """

    def __init__(self, data: Any, benchmarks: Dict[str, Callable[[Any], bool]]):
        """
        Initializes the RepresentativenessMetric with data and benchmarks.

        Parameters:
        ----------
        data : Any
            The dataset on which the representativeness metric will be calculated.
        benchmarks : Dict[str, Callable[[Any], bool]]
            A dictionary where keys describe the representativeness criteria 
            (e.g., "Age distribution") and values are callable functions 
            that validate how well the data aligns with the criteria. Each function
            should return True if the criterion is met.
        """
        super().__init__(data)
        self.benchmarks = benchmarks

    def calculate(self) -> np.float64:
        """
        Calculates the representativeness score for the dataset.

        Representativeness is computed as:
        (Number of criteria met) / (Total number of benchmarks).

        Returns:
        -------
        np.float64
            The representativeness score, where 1.0 indicates full representativeness.
        """
        # Implementation logic goes here
        pass

    def case(self) -> Generator[Any, None, None]:
        """
        Yields individual cases (e.g., demographic groups or field values) 
        that fail to meet representativeness benchmarks.

        Yields:
        ------
        Any
            Specific cases or identifiers contributing to a lower representativeness score.
        """
        # Implementation logic goes here
        pass

    def report(self) -> List[Any]:
        """
        Generates a comprehensive report of all cases that fail representativeness benchmarks.

        Returns:
        -------
        List[Any]
            A list of cases or groups that negatively impact the representativeness score.
        """
        # Implementation logic goes here
        pass

    def description(self) -> str:
        """
        Provides a description of the RepresentativenessMetric.

        Returns:
        -------
        str
            A string describing the metric and its methodology.
        """
        return (
            "RepresentativenessMetric: Evaluates how well the dataset reflects "
            "the characteristics, diversity, and proportions of the target population, "
            "based on predefined benchmarks or criteria."
        )

    def __repr__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A string representation for debugging and development purposes.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> repr(placeholder)
        "Placeholder(data=<placeholder>, missing_values=<placeholder>"
        """
        return (
            f"Placeholder(data={self.data!r}, missing_values={self.missing_values!r})"
        )

    def __str__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A simplified representation of the object's core content.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> str(placeholder)
        "Placeholder with data: <placeholder>, missing_values: <placeholder>"
        """
        return (
            f"Placeholder with data: '{self.value}', "
            f"missing_values: {self.missing_values}"
        )

class ContextualizationMetric(QualityMetric):
    """
    Measures the contextualization of the dataset, assessing how well data aligns 
    with its contextual or environmental dependencies.

    Contextualization is defined as the degree to which data adheres to contextual 
    constraints or expected relationships based on the environment, domain knowledge, 
    or external factors. It often involves evaluating interdependencies within the 
    data or between the data and external references.
    """

    def __init__(self, data: Any, context_rules: Dict[str, Callable[[Any], bool]]):
        """
        Initializes the ContextualizationMetric with data and context rules.

        Parameters:
        ----------
        data : Any
            The dataset on which the contextualization metric will be calculated.
        context_rules : Dict[str, Callable[[Any], bool]]
            A dictionary where keys describe the context (e.g., "Seasonal variation")
            and values are callable functions that evaluate adherence to the context.
            Functions should return True if the data aligns with the context.
        """
        super().__init__(data)
        self.context_rules = context_rules

    def calculate(self) -> np.float64:
        """
        Calculates the contextualization score for the dataset.

        Contextualization is computed as:
        (Number of context rules satisfied) / (Total number of context rules).

        Returns:
        -------
        np.float64
            The contextualization score, where 1.0 indicates full alignment 
            with the context.
        """
        # Implementation logic goes here
        pass

    def case(self) -> Generator[Any, None, None]:
        """
        Yields individual cases (e.g., rows, fields, or external relationships) 
        that fail to align with context rules.

        Yields:
        ------
        Any
            Specific cases or identifiers contributing to a lower contextualization score.
        """
        # Implementation logic goes here
        pass

    def report(self) -> List[Any]:
        """
        Generates a comprehensive report of all cases that fail to align 
        with the contextualization rules.

        Returns:
        -------
        List[Any]
            A list of cases or relationships that negatively impact the 
            contextualization score.
        """
        # Implementation logic goes here
        pass

    def description(self) -> str:
        """
        Provides a description of the ContextualizationMetric.

        Returns:
        -------
        str
            A string describing the metric and its methodology.
        """
        return (
            "ContextualizationMetric: Evaluates the degree to which data aligns with "
            "contextual dependencies, environmental factors, or expected relationships "
            "defined by domain-specific rules or external references."
        )

    def __repr__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A string representation for debugging and development purposes.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> repr(placeholder)
        "Placeholder(data=<placeholder>, missing_values=<placeholder>"
        """
        return (
            f"Placeholder(data={self.data!r}, missing_values={self.missing_values!r})"
        )

    def __str__(self) -> str:
        """
        Placeholder

        Returns
        -------
        str
            A simplified representation of the object's core content.

        Examples
        --------
        >>> placeholder = Placeholder(data)
        >>> placeholder.calculate()
        >>> str(placeholder)
        "Placeholder with data: <placeholder>, missing_values: <placeholder>"
        """
        return (
            f"Placeholder with data: '{self.value}', "
            f"missing_values: {self.missing_values}"
        )

