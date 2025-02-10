"""
main
----

Main entry point for the Stainalyzer tool.
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
