#!/bin/bash

brew install cmake
brew install hdf5

pip install --no-input numpy scipy pandas scikit-learn matplotlib seaborn plotly dash streamlit
pip install --no-input biopython pysam pyBigWig pybedtools
pip install --no-input opencv-python-headless scikit-image pillow pydub librosa
pip install --no-input dask joblib jupyterlab notebook optuna hyperopt
pip install --no-input faiss-cpu xgboost lightgbm transformers accelerate
pip install --no-input tensorflow keras
pip install --no-input torch torchvision torchaudio --no-cache-dir
pip install --no-input detectron2 --no-cache-dir
pip install --no-input efficientnet-pytorch face_recognition mayavi vedo pyvista trimesh open3d
pip install --no-input cupy

