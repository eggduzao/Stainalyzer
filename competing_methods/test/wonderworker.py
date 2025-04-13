"""
exceptions
----------

DimensionMetricException(Exception): Base class for all dimension metric's exception classes.
	CompletenessError(DimensionMetricException): Raised when there is an exception related to the CompletenessMetric class.
		MissingDataError(CompletenessError): Raised when a required data field is missing.
		DuplicateDataError(CompletenessError): Raised when a duplicate entry in the data is found.

	ConformanceError(DimensionMetricException): Raised when there is an exception related to the ConformanceMetric class.
		IncorrectDataFormatError (ConformanceError): Raised when a field has an incorrect format.
		InvalidDataTypeError (ConformanceError): Raised when a field has an invalid data type.
		InvalidDataValueError (ConformanceError): Raised when a field has an invalid value.
		InvalidDataUnitError (ConformanceError): Raised when a field has an invalid unit.

	ConsistencyError (DimensionMetricException): Raised when there is an exception related to the ConsistencyMetric class.
		InconsistentDataError (ConsistencyError): Raised when fields contradict other fields.

	AgreementError (DimensionMetricException): Raised when there is an exception related to the AgreementMetric class.
		IncorrectLOINCStandard (AgreementError): Raised when value or format is incongruent with the LOINC Standards.
		IncorrectICDStandard (AgreementError): Raised when value or format is incongruent with the ICD Standards.
		IncorrectHPOStandard (AgreementError): Raised when value or format is incongruent with the HPO Standards.
		IncorrectHL7FHIRStandard (AgreementError): Raised when value or format is incongruent with the HL7-FHIR Standards.

	RelevanceError (DimensionMetricException): Raised when there is an exception related to the RelevanceMetric class.
		RelevanceRuleIncongruence (RelevanceError): Raised when Relevance rules contradict other rules.

	RepresentativenessError (DimensionMetricException): Raised when there is an exception related to the RepresentativenessMetric class.
		EntropyCalculationError (RepresentativenessError): Raised when there is an error during the entropy calculation.
		RepresentativenessRuleIncongruence (RepresentativenessError): Raised when Representativeness rules contradict other rules.

	ContextualizationError (DimensionMetricException): Raised when there is an exception related to the ContextualizationMetric class.
		ContextualizationRuleIncongruence (ContextualizationError): Raised when Relevance rules contradict other rules.

YAMLConfigurationException(Exception): Base class for all data yaml-related errors.
	YAMLFileNotFoundError (YAMLConfigurationException): Raised when the YAML configuration file is missing.
	YAMLFormatError (YAMLConfigurationException): Raised when the YAML file has syntax errors or does not follow the expected schema.
	YAMLValidationError (YAMLConfigurationException): Raised when the rules defined in the YAML file contradict each other or are invalid.

OutputCreationException(Exception): Base class for all outputs - Both visualization plots and metric report.
	ReportGenerationError (OutputCreationException): Raised when there is an issue creating summary reports.
	ErrorYieldException (OutputCreationException): Raised when an error occurs inside a generator method while yielding validation errors.
	MetricComputationError (OutputCreationException): Raised when a metric cannot be computed due to an unexpected issue.
	PlotInvalidDataError (OutputCreationException): Raised when a plot cannot be created due to invalid data.
	PlotComputationError (OutputCreationException): Raised when a plot cannot be created due to computational unexpected behavior.

"""

"""
====================================================================================================
exceptions.py - General Module for I/O Processing, Analysis, and Support
====================================================================================================

Overview
--------
This module, 'exceptions.py', placeholder.


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



###############################################################################
# Constants
###############################################################################



###############################################################################
# Classes
###############################################################################

class DimensionMetricException(Exception):
    """
    Base class for all exceptions related to dimension metric calculations in the Genoma SUS project.

    This class provides methods for detailed error reporting, subtyping errors, and generating
    informative messages to aid debugging and logging.
    """
    
    def __init__(self, message, dimension=None, metric=None, severity='error'):
        """
        Initialize the DimensionMetricException.

        Parameters
        ----------
        message : str
            A descriptive error message.
        dimension : str, optional
            The name of the data quality dimension involved (e.g., completeness, consistency).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        self.message = message
        self.dimension = dimension
        self.metric = metric
        self.severity = severity
        super().__init__(self.__str__())

    def __str__(self):
        """
        Return a human-readable string representation of the exception.

        Returns
        -------
        str
            A descriptive message including dimension, metric, and severity.
        """
        details = f"DimensionMetricException: {self.message}"
        if self.dimension:
            details += f" | Dimension: {self.dimension}"
        if self.metric:
            details += f" | Metric: {self.metric}"
        details += f" | Severity: {self.severity}"
        return details

    def is_critical(self):
        """
        Determine if the exception is critical.

        Returns
        -------
        bool
            True if severity is 'critical', otherwise False.
        """
        return self.severity == 'critical'

    def report(self):
        """
        Generate a detailed report of the error.

        Returns
        -------
        dict
            A dictionary containing the exception's details.
        """
        return {
            "message": self.message,
            "dimension": self.dimension,
            "metric": self.metric,
            "severity": self.severity
        }

    def log_to_file(self, log_path="metric_errors.log"):
        """
        Log the exception details to a file.

        Parameters
        ----------
        log_path : str
            The path to the log file.
        """
        with open(log_path, 'a') as file:
            file.write(self.__str__() + '\n')

class YAMLConfigurationException(Exception):
    """
    Base class for all exceptions related to YAML configuration in the Genoma SUS project.

    This class provides functionality for validating YAML structure, reporting errors,
    and assisting in debugging configuration issues.
    """

    def __init__(self, message, file_path=None, line=None, key=None):
        """
        Initialize the YAMLConfigurationException.

        Parameters
        ----------
        message : str
            A descriptive error message.
        file_path : str, optional
            The path to the problematic YAML file.
        line : int, optional
            The line number where the error occurred.
        key : str, optional
            The YAML key associated with the error.
        """
        self.message = message
        self.file_path = file_path
        self.line = line
        self.key = key
        super().__init__(self.__str__())

    def __str__(self):
        """
        Return a human-readable string representation of the exception.

        Returns
        -------
        str
            A descriptive message including file path, line number, and key.
        """
        details = f"YAMLConfigurationException: {self.message}"
        if self.file_path:
            details += f" | File: {self.file_path}"
        if self.line:
            details += f" | Line: {self.line}"
        if self.key:
            details += f" | Key: {self.key}"
        return details

    def report(self):
        """
        Generate a detailed report of the error.

        Returns
        -------
        dict
            A dictionary containing the exception's details.
        """
        return {
            "message": self.message,
            "file_path": self.file_path,
            "line": self.line,
            "key": self.key
        }

    def log_to_file(self, log_path="yaml_errors.log"):
        """
        Log the exception details to a file.

        Parameters
        ----------
        log_path : str
            The path to the log file.
        """
        with open(log_path, 'a') as file:
            file.write(self.__str__() + '\n')

class OutputCreationException(Exception):
    """
    Base class for all exceptions related to output creation in the Genoma SUS project.

    This includes both visualization plots and metric reports, ensuring proper error
    handling for output generation processes.
    """

    def __init__(self, message, output_type=None, file_path=None, visualization_type=None):
        """
        Initialize the OutputCreationException.

        Parameters
        ----------
        message : str
            A descriptive error message.
        output_type : str, optional
            The type of output being generated ('plot', 'report').
        file_path : str, optional
            The file path where the output should be saved.
        visualization_type : str, optional
            For visualization errors, specify the chart type (e.g., 'bar', 'scatter').
        """
        self.message = message
        self.output_type = output_type
        self.file_path = file_path
        self.visualization_type = visualization_type
        super().__init__(self.__str__())

    def __str__(self):
        """
        Return a human-readable string representation of the exception.

        Returns
        -------
        str
            A descriptive message including output type, file path, and visualization type.
        """
        details = f"OutputCreationException: {self.message}"
        if self.output_type:
            details += f" | Output Type: {self.output_type}"
        if self.file_path:
            details += f" | File: {self.file_path}"
        if self.visualization_type:
            details += f" | Visualization Type: {self.visualization_type}"
        return details

    def report(self):
        """
        Generate a detailed report of the error.

        Returns
        -------
        dict
            A dictionary containing the exception's details.
        """
        return {
            "message": self.message,
            "output_type": self.output_type,
            "file_path": self.file_path,
            "visualization_type": self.visualization_type
        }

    def log_to_file(self, log_path="output_errors.log"):
        """
        Log the exception details to a file.

        Parameters
        ----------
        log_path : str
            The path to the log file.
        """
        with open(log_path, 'a') as file:
            file.write(self.__str__() + '\n')

class CompletenessError(DimensionMetricException):
    """
    Raised when there is an exception related to the CompletenessMetric class.

    This class provides methods for identifying errors that are specific to the
    completeness dimension, such as missing or duplicate data.
    """

    def __init__(self, message, dimension='completeness', metric=None, severity='error'):
        super().__init__(message, dimension, metric, severity)

class MissingDataError(CompletenessError):
    """
    Raised when a required data field is missing.

    This error helps identify cases where completeness checks fail due to missing
    critical information needed for phenotype metrics.
    """

    def __init__(self, message, missing_fields, dimension='completeness', metric=None, severity='error'):
        self.missing_fields = missing_fields
        super().__init__(message, dimension, metric, severity)

    def report_missing_fields(self):
        """
        Report the missing fields that triggered the exception.

        Returns
        -------
        list
            A list of missing fields.
        """
        return self.missing_fields

class DuplicateDataError(CompletenessError):
    """
    Raised when a duplicate entry in the data is found.

    This error identifies cases where data uniqueness is violated within the
    completeness dimension.
    """

    def __init__(self, message, duplicate_entries, dimension='completeness', metric=None, severity='error'):
        self.duplicate_entries = duplicate_entries
        super().__init__(message, dimension, metric, severity)

    def report_duplicates(self):
        """
        Report the duplicate entries that triggered the exception.

        Returns
        -------
        list
            A list of duplicate entries.
        """
        return self.duplicate_entries

class ConformanceError(DimensionMetricException):
    """
    Raised when there is an exception related to the ConformanceMetric class.

    This class provides methods for identifying errors specific to the conformance dimension,
    such as incorrect formats, invalid types, values, and units.
    """

    def __init__(self, message, dimension='conformance', metric=None, severity='error'):
        super().__init__(message, dimension, metric, severity)

class IncorrectDataFormatError(ConformanceError):
    """
    Raised when a field has an incorrect format.

    This error helps identify cases where conformance checks fail due to improperly formatted data.
    """

    def __init__(self, message, field_name, expected_format, dimension='conformance', metric=None, severity='error'):
        self.field_name = field_name
        self.expected_format = expected_format
        super().__init__(message, dimension, metric, severity)

    def report_format_issue(self):
        """
        Report the field with the incorrect format.

        Returns
        -------
        dict
            A dictionary with field name and expected format.
        """
        return {
            "field_name": self.field_name,
            "expected_format": self.expected_format
        }

class InvalidDataTypeError(ConformanceError):
    """
    Raised when a field has an invalid data type.

    This error identifies cases where data does not match the expected type.
    """

    def __init__(self, message, field_name, expected_type, actual_type, dimension='conformance', metric=None, severity='error'):
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_type = actual_type
        super().__init__(message, dimension, metric, severity)

    def report_type_mismatch(self):
        """
        Report the field with the type mismatch.

        Returns
        -------
        dict
            A dictionary with field name, expected type, and actual type.
        """
        return {
            "field_name": self.field_name,
            "expected_type": self.expected_type,
            "actual_type": self.actual_type
        }

class InvalidDataValueError(ConformanceError):
    """
    Raised when a field has an invalid value.

    This error identifies values that do not adhere to the expected value constraints.
    """

    def __init__(self, message, field_name, invalid_value, expected_values, dimension='conformance', metric=None, severity='error'):
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.expected_values = expected_values
        super().__init__(message, dimension, metric, severity)

    def report_invalid_value(self):
        """
        Report the field with the invalid value.

        Returns
        -------
        dict
            A dictionary with the field name, invalid value, and expected values.
        """
        return {
            "field_name": self.field_name,
            "invalid_value": self.invalid_value,
            "expected_values": self.expected_values
        }

class InvalidDataUnitError(ConformanceError):
    """
    Raised when a field has an invalid unit.

    This error identifies cases where data uses units that are not expected or allowed.
    """

    def __init__(self, message, field_name, invalid_unit, expected_units, dimension='conformance', metric=None, severity='error'):
        self.field_name = field_name
        self.invalid_unit = invalid_unit
        self.expected_units = expected_units
        super().__init__(message, dimension, metric, severity)

    def report_invalid_unit(self):
        """
        Report the field with the invalid unit.

        Returns
        -------
        dict
            A dictionary with the field name, invalid unit, and expected units.
        """
        return {
            "field_name": self.field_name,
            "invalid_unit": self.invalid_unit,
            "expected_units": self.expected_units
        }

class ConsistencyError(DimensionMetricException):
    """
    Raised when there is an exception related to the ConsistencyMetric class.

    This class is used to capture and handle errors related to the consistency dimension,
    which ensures that data fields do not contradict each other.
    """

    def __init__(self, message, dimension='consistency', metric=None, severity='error'):
        """
        Initialize the ConsistencyError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        dimension : str, optional
            The name of the data quality dimension ('consistency' by default).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        super().__init__(message, dimension, metric, severity)

class InconsistentDataError(ConsistencyError):
    """
    Raised when fields contradict other fields.

    This exception is triggered when the system detects logical or relational
    inconsistencies in the dataset, such as conflicting phenotype records.
    """

    def __init__(self, message, inconsistent_fields, expected_relationship, dimension='consistency', metric=None, severity='error'):
        """
        Initialize the InconsistentDataError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        inconsistent_fields : list
            A list of field names that have contradictory values.
        expected_relationship : str
            A description of the expected relationship between the fields.
        dimension : str, optional
            The name of the data quality dimension ('consistency' by default).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        self.inconsistent_fields = inconsistent_fields
        self.expected_relationship = expected_relationship
        super().__init__(message, dimension, metric, severity)

    def report_inconsistencies(self):
        """
        Generate a detailed report about the inconsistent data fields.

        Returns
        -------
        dict
            A dictionary containing information about the inconsistent fields and the expected relationship.
        """
        return {
            "inconsistent_fields": self.inconsistent_fields,
            "expected_relationship": self.expected_relationship,
            "message": self.message,
            "dimension": self.dimension,
            "metric": self.metric,
            "severity": self.severity
        }

    def log_to_file(self, log_path="consistency_errors.log"):
        """
        Log the inconsistency details to a file.

        Parameters
        ----------
        log_path : str
            The path to the log file.
        """
        with open(log_path, 'a') as file:
            file.write(f"InconsistentDataError: {self.__str__()}\n")

    def __str__(self):
        """
        Return a human-readable string representation of the inconsistent data error.

        Returns
        -------
        str
            A descriptive message including the fields involved and the expected relationship.
        """
        fields = ', '.join(self.inconsistent_fields)
        return (f"InconsistentDataError: {self.message} | Fields: {fields} | "
                f"Expected Relationship: {self.expected_relationship} | "
                f"Dimension: {self.dimension} | Metric: {self.metric} | Severity: {self.severity}")

class AgreementError(DimensionMetricException):
    """
    Raised when there is an exception related to the AgreementMetric class.

    This class handles issues related to data agreement with standardized ontologies and
    medical standards, such as LOINC, ICD, HPO, and HL7-FHIR.
    """

    def __init__(self, message, dimension='agreement', metric=None, severity='error'):
        """
        Initialize the AgreementError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        dimension : str, optional
            The name of the data quality dimension ('agreement' by default).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        super().__init__(message, dimension, metric, severity)

class IncorrectLOINCStandard(AgreementError):
    """
    Raised when a value or format is incongruent with the LOINC Standards.

    This exception is triggered when a data field does not conform to the Logical Observation
    Identifiers Names and Codes (LOINC) standards.
    """

    def __init__(self, message, field_name, invalid_value, expected_format, dimension='agreement', metric=None, severity='error'):
        """
        Initialize the IncorrectLOINCStandard exception.

        Parameters
        ----------
        message : str
            A descriptive error message.
        field_name : str
            The name of the field with the invalid value.
        invalid_value : any
            The value that does not conform to the LOINC standard.
        expected_format : str
            The expected LOINC format or code.
        dimension : str, optional
            The name of the data quality dimension ('agreement' by default).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.expected_format = expected_format
        super().__init__(message, dimension, metric, severity)

    def report_issue(self):
        """
        Generate a report of the LOINC standard violation.

        Returns
        -------
        dict
            A dictionary with the field name, invalid value, and expected format.
        """
        return {
            "field_name": self.field_name,
            "invalid_value": self.invalid_value,
            "expected_format": self.expected_format
        }

class IncorrectICDStandard(AgreementError):
    """
    Raised when a value or format is incongruent with the ICD Standards.

    This exception occurs when an ICD code does not match the expected structure or value.
    """

    def __init__(self, message, field_name, invalid_value, expected_format, dimension='agreement', metric=None, severity='error'):
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.expected_format = expected_format
        super().__init__(message, dimension, metric, severity)

    def report_issue(self):
        return {
            "field_name": self.field_name,
            "invalid_value": self.invalid_value,
            "expected_format": self.expected_format
        }

class IncorrectHPOStandard(AgreementError):
    """
    Raised when a value or format is incongruent with the HPO Standards.

    This exception indicates a mismatch against the Human Phenotype Ontology (HPO) guidelines.
    """

    def __init__(self, message, field_name, invalid_value, expected_format, dimension='agreement', metric=None, severity='error'):
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.expected_format = expected_format
        super().__init__(message, dimension, metric, severity)

    def report_issue(self):
        return {
            "field_name": self.field_name,
            "invalid_value": self.invalid_value,
            "expected_format": self.expected_format
        }

class IncorrectHL7FHIRStandard(AgreementError):
    """
    Raised when a value or format is incongruent with the HL7-FHIR Standards.

    This exception captures cases where data violates HL7 Fast Healthcare Interoperability Resources (FHIR) rules.
    """

    def __init__(self, message, field_name, invalid_value, expected_format, dimension='agreement', metric=None, severity='error'):
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.expected_format = expected_format
        super().__init__(message, dimension, metric, severity)

    def report_issue(self):
        return {
            "field_name": self.field_name,
            "invalid_value": self.invalid_value,
            "expected_format": self.expected_format
        }

class RelevanceError(DimensionMetricException):
    """
    Raised when there is an exception related to the RelevanceMetric class.

    This class is used to handle errors associated with the relevance dimension, which ensures that
    the data provided is relevant according to predefined rules and expectations.
    """

    def __init__(self, message, dimension='relevance', metric=None, severity='error'):
        """
        Initialize the RelevanceError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        dimension : str, optional
            The name of the data quality dimension ('relevance' by default).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        super().__init__(message, dimension, metric, severity)

class RelevanceRuleIncongruence(RelevanceError):
    """
    Raised when relevance rules contradict other rules.

    This exception is triggered when the relevance rules defined for a dataset are found to be
    contradictory, leading to ambiguous or invalid data relevance determinations.
    """

    def __init__(self, message, conflicting_rules, rule_details, dimension='relevance', metric=None, severity='error'):
        """
        Initialize the RelevanceRuleIncongruence exception.

        Parameters
        ----------
        message : str
            A descriptive error message.
        conflicting_rules : list
            A list of the rules that are in conflict.
        rule_details : dict
            A dictionary providing additional details about the conflicting rules.
        dimension : str, optional
            The name of the data quality dimension ('relevance' by default).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        self.conflicting_rules = conflicting_rules
        self.rule_details = rule_details
        super().__init__(message, dimension, metric, severity)

    def report_conflict(self):
        """
        Generate a detailed report about the conflicting relevance rules.

        Returns
        -------
        dict
            A dictionary containing information about the conflicting rules and their details.
        """
        return {
            "conflicting_rules": self.conflicting_rules,
            "rule_details": self.rule_details,
            "message": self.message,
            "dimension": self.dimension,
            "metric": self.metric,
            "severity": self.severity
        }

    def log_to_file(self, log_path="relevance_errors.log"):
        """
        Log the relevance rule conflict details to a file.

        Parameters
        ----------
        log_path : str
            The path to the log file.
        """
        with open(log_path, 'a') as file:
            file.write(f"RelevanceRuleIncongruence: {self.__str__()}\n")

    def __str__(self):
        """
        Return a human-readable string representation of the relevance rule incongruence.

        Returns
        -------
        str
            A descriptive message including the conflicting rules and details.
        """
        rules = ', '.join(self.conflicting_rules)
        return (f"RelevanceRuleIncongruence: {self.message} | Conflicting Rules: {rules} | "
                f"Dimension: {self.dimension} | Metric: {self.metric} | Severity: {self.severity}")

class RepresentativenessError(DimensionMetricException):
    """
    Raised when there is an exception related to the RepresentativenessMetric class.

    This class handles errors related to the representativeness dimension, ensuring that
    the data distribution accurately reflects the population of interest.
    """

    def __init__(self, message, dimension='representativeness', metric=None, severity='error'):
        """
        Initialize the RepresentativenessError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        dimension : str, optional
            The name of the data quality dimension ('representativeness' by default).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        super().__init__(message, dimension, metric, severity)

class EntropyCalculationError(RepresentativenessError):
    """
    Raised when there is an error during the entropy calculation.

    Entropy calculations are crucial for assessing representativeness by measuring
    the disorder or randomness in a dataset's distribution.
    """

    def __init__(self, message, field_name, calculation_step, dimension='representativeness', metric=None, severity='error'):
        """
        Initialize the EntropyCalculationError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        field_name : str
            The name of the field for which entropy calculation failed.
        calculation_step : str
            The specific step in the calculation where the failure occurred.
        dimension : str, optional
            The name of the data quality dimension ('representativeness' by default).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        self.field_name = field_name
        self.calculation_step = calculation_step
        super().__init__(message, dimension, metric, severity)

    def report_entropy_failure(self):
        """
        Generate a report of the entropy calculation failure.

        Returns
        -------
        dict
            A dictionary containing details about the failed entropy calculation.
        """
        return {
            "field_name": self.field_name,
            "calculation_step": self.calculation_step,
            "message": self.message,
            "dimension": self.dimension,
            "metric": self.metric,
            "severity": self.severity
        }

    def log_to_file(self, log_path="representativeness_errors.log"):
        """
        Log the entropy calculation failure details to a file.

        Parameters
        ----------
        log_path : str
            The path to the log file.
        """
        with open(log_path, 'a') as file:
            file.write(f"EntropyCalculationError: {self.__str__()}\n")

    def __str__(self):
        """
        Return a human-readable string representation of the entropy calculation error.

        Returns
        -------
        str
            A descriptive message including the field name and calculation step.
        """
        return (f"EntropyCalculationError: {self.message} | Field: {self.field_name} | "
                f"Calculation Step: {self.calculation_step} | "
                f"Dimension: {self.dimension} | Metric: {self.metric} | Severity: {self.severity}")

class RepresentativenessRuleIncongruence(RepresentativenessError):
    """
    Raised when representativeness rules contradict other rules.

    This exception is triggered when rules designed to maintain representativeness
    are found to be inconsistent with other defined rules.
    """

    def __init__(self, message, conflicting_rules, rule_details, dimension='representativeness', metric=None, severity='error'):
        """
        Initialize the RepresentativenessRuleIncongruence exception.

        Parameters
        ----------
        message : str
            A descriptive error message.
        conflicting_rules : list
            A list of the rules that are in conflict.
        rule_details : dict
            A dictionary providing additional details about the conflicting rules.
        dimension : str, optional
            The name of the data quality dimension ('representativeness' by default).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        self.conflicting_rules = conflicting_rules
        self.rule_details = rule_details
        super().__init__(message, dimension, metric, severity)

    def report_conflict(self):
        """
        Generate a detailed report about the conflicting representativeness rules.

        Returns
        -------
        dict
            A dictionary containing information about the conflicting rules and their details.
        """
        return {
            "conflicting_rules": self.conflicting_rules,
            "rule_details": self.rule_details,
            "message": self.message,
            "dimension": self.dimension,
            "metric": self.metric,
            "severity": self.severity
        }

    def log_to_file(self, log_path="representativeness_errors.log"):
        """
        Log the representativeness rule conflict details to a file.

        Parameters
        ----------
        log_path : str
            The path to the log file.
        """
        with open(log_path, 'a') as file:
            file.write(f"RepresentativenessRuleIncongruence: {self.__str__()}\n")

    def __str__(self):
        """
        Return a human-readable string representation of the representativeness rule incongruence.

        Returns
        -------
        str
            A descriptive message including the conflicting rules and details.
        """
        rules = ', '.join(self.conflicting_rules)
        return (f"RepresentativenessRuleIncongruence: {self.message} | Conflicting Rules: {rules} | "
                f"Dimension: {self.dimension} | Metric: {self.metric} | Severity: {self.severity}")

class ContextualizationError(DimensionMetricException):
    """
    Raised when there is an exception related to the ContextualizationMetric class.

    This class is used to handle errors associated with the contextualization dimension, which
    ensures that data is interpreted correctly based on its context within the dataset.
    """

    def __init__(self, message, dimension='contextualization', metric=None, severity='error'):
        """
        Initialize the ContextualizationError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        dimension : str, optional
            The name of the data quality dimension ('contextualization' by default).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        super().__init__(message, dimension, metric, severity)

class ContextualizationRuleIncongruence(ContextualizationError):
    """
    Raised when contextualization rules contradict other rules.

    This exception is triggered when the contextualization rules defined for a dataset
    conflict with each other, causing ambiguity in data interpretation.
    """

    def __init__(self, message, conflicting_rules, rule_details, dimension='contextualization', metric=None, severity='error'):
        """
        Initialize the ContextualizationRuleIncongruence exception.

        Parameters
        ----------
        message : str
            A descriptive error message.
        conflicting_rules : list
            A list of the rules that are in conflict.
        rule_details : dict
            A dictionary providing additional details about the conflicting rules.
        dimension : str, optional
            The name of the data quality dimension ('contextualization' by default).
        metric : str, optional
            The specific metric associated with the error.
        severity : str, optional
            The severity level of the error ('error', 'warning', 'critical').
        """
        self.conflicting_rules = conflicting_rules
        self.rule_details = rule_details
        super().__init__(message, dimension, metric, severity)

    def report_conflict(self):
        """
        Generate a detailed report about the conflicting contextualization rules.

        Returns
        -------
        dict
            A dictionary containing information about the conflicting rules and their details.
        """
        return {
            "conflicting_rules": self.conflicting_rules,
            "rule_details": self.rule_details,
            "message": self.message,
            "dimension": self.dimension,
            "metric": self.metric,
            "severity": self.severity
        }

    def log_to_file(self, log_path="contextualization_errors.log"):
        """
        Log the contextualization rule conflict details to a file.

        Parameters
        ----------
        log_path : str
            The path to the log file.
        """
        with open(log_path, 'a') as file:
            file.write(f"ContextualizationRuleIncongruence: {self.__str__()}\n")

    def __str__(self):
        """
        Return a human-readable string representation of the contextualization rule incongruence.

        Returns
        -------
        str
            A descriptive message including the conflicting rules and details.
        """
        rules = ', '.join(self.conflicting_rules)
        return (f"ContextualizationRuleIncongruence: {self.message} | Conflicting Rules: {rules} | "
                f"Dimension: {self.dimension} | Metric: {self.metric} | Severity: {self.severity}")

class YAMLFileNotFoundError(YAMLConfigurationException):
    """
    Raised when the YAML configuration file is missing.

    This exception is triggered when the system attempts to load a YAML file that does not exist.
    """

    def __init__(self, message, file_path, severity='critical'):
        """
        Initialize the YAMLFileNotFoundError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        file_path : str
            The path to the missing YAML file.
        severity : str, optional
            The severity level of the error ('critical' by default).
        """
        self.severity = severity
        super().__init__(message, file_path=file_path)

    def is_critical(self):
        """
        Determine if the exception is critical.

        Returns
        -------
        bool
            True if severity is 'critical', otherwise False.
        """
        return self.severity == 'critical'

class YAMLFormatError(YAMLConfigurationException):
    """
    Raised when the YAML file has syntax errors or does not follow the expected schema.

    This exception is triggered when the YAML parser encounters structural or syntactic
    inconsistencies in the configuration file.
    """

    def __init__(self, message, file_path, line, problematic_content, severity='error'):
        """
        Initialize the YAMLFormatError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        file_path : str
            The path to the YAML file with the format error.
        line : int
            The line number where the format issue occurred.
        problematic_content : str
            The content that caused the format error.
        severity : str, optional
            The severity level of the error ('error' by default).
        """
        self.problematic_content = problematic_content
        self.severity = severity
        super().__init__(message, file_path=file_path, line=line)

    def report_format_issue(self):
        """
        Generate a report of the YAML format error.

        Returns
        -------
        dict
            A dictionary with details about the format error.
        """
        return {
            "message": self.message,
            "file_path": self.file_path,
            "line": self.line,
            "problematic_content": self.problematic_content,
            "severity": self.severity
        }

class YAMLValidationError(YAMLConfigurationException):
    """
    Raised when the rules defined in the YAML file contradict each other or are invalid.

    This exception is triggered when the validation logic detects logical inconsistencies
    or invalid configurations in the YAML rules.
    """

    def __init__(self, message, file_path, rule_name, conflicting_values, severity='error'):
        """
        Initialize the YAMLValidationError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        file_path : str
            The path to the YAML file with the invalid rule.
        rule_name : str
            The name of the rule that caused the validation error.
        conflicting_values : list
            A list of values that are in conflict.
        severity : str, optional
            The severity level of the error ('error' by default).
        """
        self.rule_name = rule_name
        self.conflicting_values = conflicting_values
        self.severity = severity
        super().__init__(message, file_path=file_path)

    def report_validation_issue(self):
        """
        Generate a report of the YAML validation error.

        Returns
        -------
        dict
            A dictionary with details about the validation error.
        """
        return {
            "message": self.message,
            "file_path": self.file_path,
            "rule_name": self.rule_name,
            "conflicting_values": self.conflicting_values,
            "severity": self.severity
        }

class YAMLFieldNotFound(YAMLConfigurationException):
    """
    Raised when the name of the column does not match any field in the YAML file.

    This exception is triggered when the validation logic detects that a table's field name
    does not match any field name configurations in the YAML rules.
    """

    def __init__(self, message, file_path, rule_name, conflicting_table_field, severity='error'):
        """
        Initialize the YAMLValidationError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        file_path : str
            The path to the YAML file with the invalid rule.
        rule_name : str
            The name of the rule that caused the validation error.
        conflicting_table_field : str
            The name of the conflicting field in the table.
        severity : str, optional
            The severity level of the error ('error' by default).
        """
        self.rule_name = rule_name
        self.conflicting_table_field = conflicting_table_field
        self.severity = severity
        super().__init__(message, file_path=file_path)

    def report_validation_issue(self):
        """
        Generate a report of the YAML field not found error.

        Returns
        -------
        dict
            A dictionary with details about YAML field not found error.
        """
        return {
            "message": self.message,
            "file_path": self.file_path,
            "rule_name": self.rule_name,
            "conflicting_table_field": self.conflicting_table_field,
            "severity": self.severity
        }

class ReportGenerationError(OutputCreationException):
    """
    Raised when there is an issue creating summary reports.

    This exception captures errors that occur while generating reports, such as missing
    data, formatting issues, or write permission problems.
    """

    def __init__(self, message, report_name, file_path, severity='error'):
        """
        Initialize the ReportGenerationError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        report_name : str
            The name of the report that failed to generate.
        file_path : str
            The file path where the report was supposed to be saved.
        severity : str, optional
            The severity level of the error ('error' by default).
        """
        self.report_name = report_name
        self.severity = severity
        super().__init__(message, output_type='report', file_path=file_path)

    def report_failure(self):
        """
        Generate a report of the report generation failure.

        Returns
        -------
        dict
            A dictionary with details about the failed report generation.
        """
        return {
            "message": self.message,
            "report_name": self.report_name,
            "file_path": self.file_path,
            "severity": self.severity
        }

class ErrorYieldException(OutputCreationException):
    """
    Raised when an error occurs inside a generator method while yielding validation errors.

    This exception occurs when a generator fails to yield errors correctly, potentially
    due to logical issues or resource exhaustion.
    """

    def __init__(self, message, generator_name, problematic_value, severity='error'):
        """
        Initialize the ErrorYieldException.

        Parameters
        ----------
        message : str
            A descriptive error message.
        generator_name : str
            The name of the generator where the error occurred.
        problematic_value : any
            The value that caused the generator to fail.
        severity : str, optional
            The severity level of the error ('error' by default).
        """
        self.generator_name = generator_name
        self.problematic_value = problematic_value
        self.severity = severity
        super().__init__(message, output_type='generator')

    def report_failure(self):
        """
        Generate a report of the generator failure.

        Returns
        -------
        dict
            A dictionary with details about the generator error.
        """
        return {
            "message": self.message,
            "generator_name": self.generator_name,
            "problematic_value": self.problematic_value,
            "severity": self.severity
        }

class MetricComputationError(OutputCreationException):
    """
    Raised when a metric cannot be computed due to an unexpected issue.

    This exception captures cases where metric computations fail due to issues like
    missing values, division by zero, or numerical instability.
    """

    def __init__(self, message, metric_name, computation_step, severity='error'):
        """
        Initialize the MetricComputationError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        metric_name : str
            The name of the metric that failed to compute.
        computation_step : str
            The step during which the computation failed.
        severity : str, optional
            The severity level of the error ('error' by default).
        """
        self.metric_name = metric_name
        self.computation_step = computation_step
        self.severity = severity
        super().__init__(message, output_type='metric')

    def report_failure(self):
        """
        Generate a report of the metric computation failure.

        Returns
        -------
        dict
            A dictionary with details about the failed metric computation.
        """
        return {
            "message": self.message,
            "metric_name": self.metric_name,
            "computation_step": self.computation_step,
            "severity": self.severity
        }

class PlotInvalidDataError(OutputCreationException):
    """
    Raised when a plot cannot be created due to invalid data.

    This exception indicates that the dataset provided for a visualization contains
    invalid or incompatible values.
    """

    def __init__(self, message, plot_type, invalid_data_summary, severity='error'):
        """
        Initialize the PlotInvalidDataError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        plot_type : str
            The type of plot being created (e.g., 'bar', 'scatter').
        invalid_data_summary : str
            A brief summary of the data issues encountered.
        severity : str, optional
            The severity level of the error ('error' by default).
        """
        self.plot_type = plot_type
        self.invalid_data_summary = invalid_data_summary
        self.severity = severity
        super().__init__(message, output_type='plot')

    def report_failure(self):
        """
        Generate a report of the plot data validation failure.

        Returns
        -------
        dict
            A dictionary with details about the invalid data and plot type.
        """
        return {
            "message": self.message,
            "plot_type": self.plot_type,
            "invalid_data_summary": self.invalid_data_summary,
            "severity": self.severity
        }

class PlotComputationError(OutputCreationException):
    """
    Raised when a plot cannot be created due to computational unexpected behavior.

    This exception captures cases where the underlying computation for a plot fails
    because of numerical issues or software errors.
    """

    def __init__(self, message, plot_type, computation_step, severity='error'):
        """
        Initialize the PlotComputationError.

        Parameters
        ----------
        message : str
            A descriptive error message.
        plot_type : str
            The type of plot being created.
        computation_step : str
            The step during which the computation failed.
        severity : str, optional
            The severity level of the error ('error' by default).
        """
        self.plot_type = plot_type
        self.computation_step = computation_step
        self.severity = severity
        super().__init__(message, output_type='plot')

    def report_failure(self):
        """
        Generate a report of the plot computation failure.

        Returns
        -------
        dict
            A dictionary with details about the failed plot computation.
        """
        return {
            "message": self.message,
            "plot_type": self.plot_type,
            "computation_step": self.computation_step,
            "severity": self.severity
        }

