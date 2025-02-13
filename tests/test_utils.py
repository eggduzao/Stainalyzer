import subprocess
import os

def check_and_install_apt(package):
    """Check if an APT package is installed, and install it rootlessly if possible."""
    try:
        # Check if the package is installed
        result = subprocess.run(["dpkg", "-s", package], capture_output=True, text=True)
        if "Status: install ok installed" in result.stdout:
            return f"{package}: Already Installed!"
        
        # Try installing via apt-get rootless
        install_cmd = ["apt-get", "download", package]
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return f"{package}: Success!"
        else:
            return f"{package}: Could not install rootless. See log below.\n{result.stderr}"
    except Exception as e:
        return f"{package}: Do not exist! See log below.\n{str(e)}"

def check_and_install_conda(package):
    """Check if a Conda package is installed and install it if not."""
    try:
        # Check if the package exists in Conda
        result = subprocess.run(["conda", "list", package], capture_output=True, text=True)
        if package in result.stdout:
            return f"{package}: Already Installed!"
        
        # Try installing via Conda
        install_cmd = ["conda", "install", "-n", "base", "-c", "conda-forge", package, "-y"]
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return f"{package}: Success!"
        else:
            return f"{package}: Do not exist! See log below.\n{result.stderr}"
    except Exception as e:
        return f"{package}: Do not exist! See log below.\n{str(e)}"

# List of APT and Conda packages
apt_packages = [
    "build-essential", "wget", "curl", "git", "bedtools", "default-jdk", "ffmpeg", "gdal-bin", "gfortran",
    "graphviz", "htslib", "libatlas-base-dev", "libavcodec-dev", "libavformat-dev", "libbz2-dev",
    "libcurl4-openssl-dev", "libdlib-dev", "libffi-dev", "libgeos-dev", "libgl1-mesa-glx", "libglib2.0-0",
    "libgsl0-dev", "libhdf5-dev", "libheif-dev", "libjpeg-dev", "liblapack-dev", "liblzma-dev",
    "libncurses5-dev", "libncursesw5-dev", "libnetcdf-dev", "libopenblas-dev", "libopencv-dev",
    "libpng-dev", "libproj-dev", "libreadline-dev", "libsm6", "libsndfile1", "libsqlite3-dev",
    "libssl-dev", "libswscale-dev", "libtiff-dev", "libv4l-dev", "libxext6", "libxml2-dev",
    "libxrender-dev", "libxslt1-dev", "openjdk-11-jdk", "python3-pip", "r-base", "samtools",
    "zlib1g-dev", "imagemagick", "sox"
]

conda_packages = [
    "numpy", "scipy", "pandas", "scikit-learn", "tensorflow", "pytorch", "torchvision", "torchaudio",
    "matplotlib", "seaborn", "plotly", "dash", "streamlit", "biopython", "pysam", "pyBigWig", "pybedtools",
    "opencv", "scikit-image", "pillow", "pydub", "librosa", "dask", "joblib", "jupyterlab", "notebook",
    "optuna", "hyperopt", "faiss-cpu", "modin[all]", "statsmodels", "cvxpy", "category_encoders",
    "featuretools", "protobuf", "py7zr", "pymzml", "pyteomics", "pyvips", "mayavi", "vedo", "pyvista",
    "bcftools", "samtools", "open3d", "trimesh",
    "HTSeq", "SimpleITK", "albumentations", "altair", "anndata", "yellowbrick", "annoy", "apex", "av", 
    "bambi", "bayesian-optimization", "bioblend", "biopython", "biopython-extras", "bitstring", "catboost", 
    "category_encoders", "cobra", "cvxpy", "dagster", "dash", "dask", "datashader", "datatable", 
    "deeplearning4j", "detectron2", "duckdb", "dvc", "efficientnet-pytorch", "eli5", "face_recognition", 
    "faiss-cpu", "fastai", "feature-engine", "featuretools", "ffmpeg-python", "fuzzywuzzy", "geopandas", 
    "gffutils", "gplearn", "great_expectations", "gym", "h2o", "h5py", "holoviews", "horovod", 
    "huggingface-hub", "hydra-core", "hyperopt", "imageio", "imbalanced-learn", "imgaug", "joblib", 
    "jupyterlab", "kedro", "keras", "keras-tuner", "librosa", "lightgbm", "lime", "luigi", "mahotas", 
    "matplotlib>=3.3.0", "methylpy", "missingno", "zarr", "ml-dtypes", "mleap", "mlflow", "mlxtend", 
    "modin[all]", "moviepy", "multiqc", "mxnet", "netCDF4", "numpy>=1.18.0", "onnx", "open3d", 
    "opencv-python", "openpyxl", "optuna", "pandas", "pandera", "patsy", "pillow", "plotly", 
    "polars", "protobuf", "py7zr", "pyBigWig", "pyaudio", "pybedtools", "pybzip2", "pycaret", 
    "pycocotools", "pydantic", "pydicom", "pydot", "pyensembl", "pyheif", "pyjanitor", "pymc3", 
    "pymetagenomics", "pymzml", "pysam", "pyspark", "pyteomics", "pytesseract", "python-dateutil", 
    "pytorch-lightning", "pytz", "pyvips", "pywavelets", "pyyaml", "ray", "requests", "rna-tools", 
    "rpy2", "scanpy", "scikit-bio", "scikit-image", "scikit-learn>=0.23.0", "scikit-optimize", 
    "scipy", "seaborn", "shap", "skorch", "snakemake", "soundfile", "sqlalchemy", 
    "stylegan3", "sympy", "tabulate", "tensorboard", "tensorboardX", "tensorflow", "tifffile", 
    "torch>=1.7.0", "torchaudio", "torchvision", "tpot", "tqdm", "transformers", "vaex", 
    "vcfpy", "xgboost", "xlrd", "xlwt", "ydata-profiling"
]

# Logs
install_log = []
print("Installing APT packages...")
for pkg in apt_packages:
    result = check_and_install_apt(pkg)
    print(result)
    install_log.append(result)

print("\nInstalling Conda packages...")
for pkg in conda_packages:
    result = check_and_install_conda(pkg)
    print(result)
    install_log.append(result)

# Save logs to file
log_filename = "install_log.txt"
with open(log_filename, "w") as log_file:
    log_file.write("\n".join(install_log))

print(f"\nInstallation complete. Log saved to {log_filename}")
