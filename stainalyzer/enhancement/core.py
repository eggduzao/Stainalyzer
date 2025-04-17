"""
core
----

Parses command-line arguments and performs core processing functions for the Stainalyzer tool.
"""

############################################################################################################
### Import
############################################################################################################

import argparse

from stainalyzer import __version__
from stainalyzer.enhancement.preprocessor import EnhancementPreprocessor
from stainalyzer.enhancement.trainer import EnhancementTrainer
from stainalyzer.enhancement.tester import EnhancementTester

############################################################################################################
### Constants
############################################################################################################

############################################################################################################
### Argument Parsing
############################################################################################################

def parse_args():
    """
    Parses command-line arguments for the Stainalyzer tool.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments.

    Positional Arguments
    --------------------
    input_training_folder : str
        Path to the input training folder.
    output_training_folder : str
        Path to the output training folder where results will be saved.
    root_name : str
        Root directory name used to structure the file paths.

    Optional Arguments
    ------------------
    --severity : float, optional
        Specifies the training severity. Default is 1.0.

    Standard Options
    ----------------
    -h, --help : Show help message and exit.
    -v, --version : Show the version of the Stainalyzer tool and exit.

    Notes
    -----
    - Ensure that input paths are valid directories.
    - The tool assumes images are formatted correctly within the input directory.
    
    """
    parser = argparse.ArgumentParser(
        description="Stainalyzer Tool: A robust tool for DAB-stained image analysis in microscopy.",
        epilog="Example Usage: Stainalyzer /path/to/input/root_folder/ /path/to/output root_folder --severity 0.5"
    )

    # Positional arguments
    parser.add_argument(
        "input_training_folder",
        type=str,
        help="Path to the input training folder."
    )
    parser.add_argument(
        "output_training_folder",
        type=str,
        help="Path to the output training folder where results will be saved."
    )
    parser.add_argument(
        "root_name",
        type=str,
        help="Root directory name used to structure the file paths."
    )

    # Optional arguments
    parser.add_argument(
        "--severity",
        type=float,
        default=0.5,
        help="Specifies the training severity from 0.0 (increase false positives) to 1.0 (decrease true positives) (default: 0.5)."
    )

    # Version and Help
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f'Stainalyzer {__version__}',
        help="Show program version and exit."
    )

    args = parser.parse_args()

    # Validate arguments
    if args.severity < 0.0 or args.severity > 1.0:
        parser.error("--severity must be between 0.0 and 1.0.")

    return args

############################################################################################################
### Core Functions
############################################################################################################

def core_function(input_training_folder, output_training_folder, root_name, training_severity=1.0):
    """
    Core processing function for the Stainalyzer tool.

    This function initializes the `Trainer` class and starts the training process 
    based on the provided directories and severity level.

    Parameters
    ----------
    input_training_folder : str
        Path to the input training folder.
    output_training_folder : str
        Path to the output training folder where results will be saved.
    root_name : str
        Root directory name used to structure the file paths.
    training_severity : float, optional
        Specifies the training severity (default is 1.0).

    Raises
    ------
    FileNotFoundError
        If the input training folder does not exist.
    ValueError
        If invalid parameters are provided.
    
    Notes
    -----
    The actual training logic is handled by the `Trainer` class, which processes
    images and generates output in the specified directory.

    """
    try:
        trainer = Trainer(
            training_image_path=input_training_folder,
            severity=training_severity,
            root_name=root_name
        )
        trainer.train(output_location=output_training_folder)
    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except ValueError as val_error:
        print(f"Error: {val_error}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

