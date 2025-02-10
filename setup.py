
# Stainalyzer Package

import os
import re
from setuptools import setup, find_packages  # setuptools is the standard tool for packaging Python projects.

def read_version():
    version_file = os.path.join(os.path.dirname(__file__), 'Stainalyzer', '__init__.py')
    with open(version_file, 'r') as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

# setup() is the function that handles all the package metadata and configuration.
setup(
    # Name of your package. This should be unique and is how it will be identified on PyPI.
    name='Stainalyzer',  

    # Version of your package. Follows semantic versioning (e.g., MAJOR.MINOR.PATCH).
    version=read_version(),  

    # A brief, one-line description of what your package does.
    description='A self-supervised learning toolkit for histopathology microscopy image analysis.',  

    # A longer description of your project. Often pulled from your README file.
    long_description=open('README.md').read(),  
    
    # Specify the format of your long description. Common options: 'text/markdown' or 'text/x-rst'.
    long_description_content_type='text/markdown',  

    # Author's name.
    author='Eduardo Gade Gusmao',  

    # Author's email.
    author_email='eduardo.gusmao@tum-sls.de',  

    # URL for the project, e.g., GitHub repository.
    url='https://github.com/eggduzao/Stainalyzer',  

    # List of keywords to help people find your package when searching PyPI.
    keywords='DAB, immunohistopathology, self-supervised learning, image analysis, deep learning',  

    # Automatically find all packages in the current directory, excluding certain patterns.
    packages=find_packages(exclude=['dev', 'test', 'tests', 'docs', 'data', 'examples']),  

    # Include additional files specified in MANIFEST.in.
    include_package_data=True,  

	# List of dependencies that will be installed when someone installs your package.
	install_requires=[
	    'numpy>=1.18.0,<2.1.0',       	# For numerical computations, compatible with numpy 2.0.2
	    'scipy>=1.5.0,<1.14.0',       	# Scientific computing library, compatible with scipy 1.13.1
	    'torch>=1.7.0,<2.7.0',        	# PyTorch for deep learning, compatible with torch 2.6.0
	    'torchvision>=0.8.0,<0.22.0', 	# Image transformations and datasets for PyTorch, compatible with torchvision 0.21.0
	    'pandas>=1.0.0,<2.3.0',       	# Data manipulation and analysis, compatible with pandas 2.2.3
	    'opencv-python>=4.2.0,<4.11.0', # Image processing, compatible with OpenCV 4.10.0
	    'openpyxl>=3.0.0,<3.2.0',     	# Reading and writing Excel files, compatible with openpyxl 3.1.5
	    'seaborn>=0.11.0,<0.14.0',    	# Data visualization, compatible with seaborn 0.13.2
	    'Pillow>=8.0.0,<10.5.0',      	# Python Imaging Library, compatible with Pillow 10.4.0
	    'matplotlib>=3.3.0,<3.10.0',  	# Plotting and visualization, compatible with matplotlib 3.9.2
	    'scikit-learn>=0.23.0,<1.6.0' 	# Machine learning utilities, compatible with scikit-learn 1.5.2
	],

    # Additional groups of dependencies, e.g., for development or testing.
	extras_require={

	    # Development tools for maintaining code quality, formatting, and versioning
	    'dev': [
	        'black>=23.0',            # Automatic code formatter for consistent style
	        'flake8>=6.0',            # Linter for identifying code issues and enforcing PEP8
	        'isort>=5.12',            # Automatically sort and organize imports
	        'pre-commit>=3.3',        # Manage and run pre-commit hooks for code checks
	        'bump2version>=1.0',      # Automate version bumping for releases
	        'mypy>=1.5',              # Optional static type checker for Python
	        'tox>=4.0',               # Test automation across multiple environments (can overlap with testing)
	        'pylint>=3.0'             # Code analysis tool for identifying potential issues
	    ],

	    # Testing tools for running, managing, and measuring test coverage
	    'test': [
	        'pytest>=7.0',            # Framework for writing and running unit tests
	        'pytest-cov>=4.0',        # Plugin for measuring code coverage with pytest
	        'coverage>=7.2',          # Track which parts of the code are covered by tests
	        'hypothesis>=6.0',        # Property-based testing, generates edge cases automatically
	        'tox>=4.0',               # Automate testing across multiple environments (also useful in dev)
	        'faker>=18.0',            # Generate fake data for testing
	        'responses>=0.23'         # Mock out HTTP responses in tests
	    ],

	    # Documentation tools for generating and managing project documentation
	    'docs': [
	        'sphinx>=7.0',            # Powerful documentation generator
	        'sphinx-rtd-theme>=1.3',  # ReadTheDocs theme for Sphinx
	        'sphinx-autodoc-typehints>=1.23',  # Automatically include type hints in docs
	        'myst-parser>=2.0',       # Support for Markdown in Sphinx
	        'nbsphinx>=0.9',          # Integrate Jupyter Notebooks into documentation
	        'mkdocs>=1.5',            # Alternative static site generator for docs
	        'mkdocs-material>=9.0'    # Popular theme for MkDocs
	    ]
	},

	# Fine-grained control of what to include in addition
	package_data={
	    'Stainalyzer': [
	        'data/colornames.txt',          # Example datasets in CSV format
	        'data/DAB_Training',			# Sample training images
	        'data/DAB_Training_Output',     # Sample training images
	        'datapath.py',					# Script to clean the DAB-Project data tree
	        'clean.sh',           			# Script to clean MAC OS X trash files
	        'rsync.sh'           			# Script to convert data between cluster and local
	    ]
	},

    # Specifies the license under which your package is distributed.
    license='MIT',  

    # Classifiers provide metadata about your package for PyPI and users.
    classifiers=[

    	# Official Stage
        'Development Status :: 3 - Alpha',  # Indicates maturity level of the project

        # Target audience
        'Intended Audience :: Science/Research',

	    # Topics
	    'Topic :: Scientific/Engineering :: Bio-Informatics',
	    'Topic :: Scientific/Engineering :: Image Recognition',
	    'Topic :: Scientific/Engineering :: Artificial Intelligence',
	    'Topic :: Scientific/Engineering :: Artificial Intelligence :: Deep Learning',
	    'Topic :: Scientific/Engineering :: Mathematics',
	    'Topic :: Scientific/Engineering :: Mathematics :: Probability and Statistics',
	    'Topic :: Scientific/Engineering :: Medical Science Apps.',
	    'Topic :: Scientific/Engineering :: Visualization',

	    # License Type
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Supported Python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],

	# Official Stages:
	# Status					Code	Description
	# 1 - Planning				1		Initial planning, no functional code yet.
	# 2 - Pre-Alpha				2		Early development, incomplete and unstable.
	# 3 - Alpha					3		Limited functionality, experimental, may be unstable.
	# 4 - Beta					4		Feature-complete but may still have bugs, open for testing.
	# 5 - Production/Stable		5		Stable release, suitable for general use.
	# 6 - Mature				6		Fully mature, unlikely to change significantly.
	# 7 - Inactive				7		No longer maintained or supported.

    # This specifies the minimum Python version required.
    python_requires='>=3.7',  

    # Entry points define executable scripts that can be run from the command line.
    entry_points={
        'console_scripts': [
            'Stainalyzer=Stainalyzer.main:main'  # Creates a command 'Stainalyzer' to run 'main()'' in 'Stainalyzer/Stainalyzer/main.py'
        ]
    },

    # Project-related URLs.
    project_urls={
        'Source': 'https://github.com/yourusername/microscopy'
    },
)
