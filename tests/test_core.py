import subprocess

# APT packages to check
apt_packages = [
    "build-essential", "wget", "curl", "git", "bedtools", "default-jdk", "ffmpeg", "gdal-bin", "gfortran", 
    "graphviz", "htslib", "libatlas-base-dev", "libavcodec-dev", "libavformat-dev", "libbz2-dev", 
    "libcurl4-openssl-dev", "libdlib-dev", "libffi-dev", "libgeos-dev", "libgl1-mesa-glx", "libglib2.0-0", 
    "libgsl0-dev", "libhdf5-dev", "libheif-dev", "libjpeg-dev", "liblapack-dev", "liblzma-dev", 
    "libncurses5-dev", "libncursesw5-dev", "libnetcdf-dev", "libopenblas-dev", "libopencv-dev", 
    "libpng-dev", "libproj-dev", "libreadline-dev", "libsm6", "libsndfile1", "libsqlite3-dev", 
    "libssl-dev", "libswscale-dev", "libtiff-dev", "libv4l-dev", "libxext6", "libxml2-dev", 
    "libxrender-dev", "libxslt1-dev", "openjdk-11-jdk", "python3.12", "python3.12-venv", 
    "python3.12-dev", "python3-pip", "r-base", "samtools", "zlib1g-dev"
]

# Conda packages to check
conda_packages = [
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
    "opencv-python", "openpyxl", "optuna", "pandas>=1.0.0", "pandera", "patsy", "pillow", "plotly", 
    "polars", "protobuf", "py7zr", "pyBigWig", "pyaudio", "pybedtools", "pybzip2", "pycaret", 
    "pycocotools", "pydantic", "pydicom", "pydot", "pyensembl", "pyheif", "pyjanitor", "pymc3", 
    "pymetagenomics", "pymzml", "pysam", "pyspark", "pyteomics", "pytesseract", "python-dateutil", 
    "pytorch-lightning", "pytz", "pyvips", "pywavelets", "pyyaml", "ray", "requests", "rna-tools", 
    "rpy2", "scanpy", "scikit-bio", "scikit-image", "scikit-learn>=0.23.0", "scikit-optimize", 
    "scipy>=1.5.0", "seaborn>=0.11.0", "shap", "skorch", "snakemake", "soundfile", "sqlalchemy", 
    "stylegan3", "sympy", "tabulate", "tensorboard", "tensorboardX", "tensorflow", "tifffile", 
    "torch>=1.7.0", "torchaudio", "torchvision>=0.8.0", "tpot", "tqdm", "transformers", "vaex", 
    "vcfpy", "xgboost", "xlrd", "xlwt", "ydata-profiling"
]

def check_apt_packages(packages):
    print("Checking APT packages...")
    for package in packages:
        try:
            result = subprocess.run(["apt-cache", "search", package], capture_output=True, text=True)
            if result.stdout.strip():
                print(f"APT Package Found: {package}")
            else:
                print(f"APT Package Not Found: {package}")
        except Exception as e:
            print(f"Error checking APT package {package}: {e}")

def check_conda_packages(packages):
    print("\nChecking Conda packages...")
    for package in packages:
        try:
            result = subprocess.run(["conda", "search", package], capture_output=True, text=True)
            if result.stdout.strip():
                print(f"Conda Package Found: {package}")
            else:
                print(f"Conda Package Not Found: {package}")
        except Exception as e:
            print(f"Error checking Conda package {package}: {e}")

if __name__ == "__main__":
    check_apt_packages(apt_packages)
    check_conda_packages(conda_packages)
