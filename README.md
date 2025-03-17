# Stainalyzer
### Open and publicly available tool to automatically evaluate immunohistochemically-stained antibodies

## Introduction

Welcome to **Stainalyzer**, an advanced and efficient tool designed to analyze stains and regions of interest in microscopy images. Stainalyzer is particularly useful for **histology** (brightfield images), **fluorescence microscopy** (FISH), and **confocal imaging**. This tool leverages state-of-the-art image processing techniques to detect and quantify stains, assisting researchers and clinicians in their analyses.

This repository serves as the official source code for Stainalyzer, providing comprehensive documentation, installation instructions, usage guidelines, and development notes.

---

## Features

Stainalyzer comes packed with a variety of features that make it an essential tool for image analysis in biomedical research. Some of the key features include:

- **Automated stain detection**: Detects and isolates stained regions with high accuracy.
- **Multi-channel image support**: Works with brightfield, fluorescence, and confocal images.
- **Segmentation algorithms**: Supports thresholding, clustering, and deep learning-based segmentation.
- **Quantification metrics**: Measures area, intensity, and distribution of stains.
- **Batch processing**: Analyzes multiple images simultaneously for high-throughput workflows.
- **Support for multiple formats**: Reads and writes common image formats, including TIFF, PNG, JPEG, and specialized microscopy formats.
- **User-friendly interface**: Simple command-line usage with optional GUI support.

---

## Installation

Stainalyzer is designed to be easy to install and run on **Linux, macOS, and Windows**. The installation process follows the standard Python package setup. To install Stainalyzer, follow the steps below:

### Prerequisites
Before installing Stainalyzer, ensure you have the following dependencies installed:

- **Python 3.8+** (Recommended: latest stable release)
- **pip** (Python package manager)
- **Virtual environment (optional but recommended)**

Additionally, Stainalyzer requires the following Python libraries:

- **numpy** (Numerical computing)
- **scipy** (Scientific computing)
- **opencv-python** (Image processing)
- **scikit-image** (Image analysis tools)
- **matplotlib** (Visualization)
- **pandas** (Data handling)
- **tqdm** (Progress bars)

### Step-by-Step Installation Guide

#### 1. Clone the Repository
To get started, clone this repository from GitHub:

```bash
$ git clone https://github.com/yourusername/stainalyzer.git
$ cd stainalyzer
```

#### 2. Create a Virtual Environment (Recommended)
To keep dependencies isolated, it is recommended to create a virtual environment:

```bash
$ python -m venv venv
$ source venv/bin/activate  # On macOS/Linux
$ venv\Scripts\activate    # On Windows
```

#### 3. Install Dependencies
Use `pip` to install the required dependencies:

```bash
$ pip install -r requirements.txt
```

If you do not have `requirements.txt`, you can manually install dependencies:

```bash
$ pip install numpy scipy opencv-python scikit-image matplotlib pandas tqdm
```

#### 4. Verify Installation
Once installed, verify that Stainalyzer is working correctly:

```bash
$ python stainalyzer.py --help
```

If the help message appears, the installation was successful.

---

## Usage Guide

Stainalyzer is designed to be user-friendly and flexible. Below are basic commands to analyze images.

### Running Stainalyzer
To analyze a single image, use:

```bash
$ python stainalyzer.py --input image.tiff --output results.csv
```

### Batch Processing
To analyze multiple images in a directory:

```bash
$ python stainalyzer.py --input images/ --output results/
```

### Adjusting Parameters
Customize stain detection with additional parameters:

```bash
$ python stainalyzer.py --input image.tiff --output results.csv --threshold 0.5 --method kmeans
```

For a full list of options:

```bash
$ python stainalyzer.py --help
```

---

## Methodology

Stainalyzer uses the GhostNet Neural Network. The basic fundaments is as follows.

### Latent Stochastic Process

We define the generative process as a **stochastic differential equation (SDE)**:

The stochastic differential equation governing the process is:

```math
d\mathbf{z}_t = \mu(\mathbf{z}_t, t) dt + \sigma(\mathbf{z}_t, t) d\mathbf{W}_t,
```

where:

```math
\mathbf{z}_t
```
is the latent representation at time \( t \), 

```math
\mu(\mathbf{z}_t, t)
```
is the drift term,

```math
\sigma(\mathbf{z}_t, t) ,
``` 

is the diffusion coefficient, and 

```math
\mathbf{W}_t ,
```

is a Wiener process modeling Brownian motion.


The transition density of the process is governed by the Fokker-Planck equation:

```math
\frac{\partial p(\mathbf{z}, t)}{\partial t} = - \nabla \cdot (\mu(\mathbf{z}, t) p(\mathbf{z}, t)) + \frac{1}{2} \nabla^2 (\sigma^2(\mathbf{z}, t) p(\mathbf{z}, t)).
```

### Variational Inference

The encoder network approximates the posterior \( q_\phi(\mathbf{z} | \mathbf{x}) \) via the **reparameterization trick**:

```math
\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).
```

The evidence lower bound (ELBO) is:

```math
\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} \left[ \log p_\theta(\mathbf{x} | \mathbf{z}) \right] - D_{KL} \left( q_\phi(\mathbf{z} | \mathbf{x}) || p(\mathbf{z}) \right).
```

### Diffusion Model Integration

The forward process gradually adds Gaussian noise to the latent space:

```math
q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}(\mathbf{z}_t; \alpha_t \mathbf{z}_0, \sigma_t^2 I),
```

where \( \alpha_t \) and \( \sigma_t \) define the noise schedule. The reverse process learns to denoise step-by-step:

```math
p_\theta(\mathbf{z}_{t-1} | \mathbf{z}_t) = \mathcal{N}(\mathbf{z}_{t-1}; \mu_\theta(\mathbf{z}_t, t), \Sigma_\theta(\mathbf{z}_t, t)).
```

### Training and Optimization

We minimize the variational loss function:

```math
\mathcal{L}_{\text{total}} = \lambda_{\text{VAE}} \mathcal{L}(\theta, \phi) + \lambda_{\text{diff}} \mathcal{L}_{\text{diffusion}},
```

where the diffusion loss is given by:

```math
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, \mathbf{z}_t} \left[ || \mathbf{z}_t - \mu_\theta(\mathbf{z}_t, t) ||^2 \right].
```

---

## Development & Contribution

We welcome contributions from the community! If you’d like to contribute to Stainalyzer, please follow these guidelines:

### Setting Up a Development Environment

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   $ git clone https://github.com/yourusername/stainalyzer.git
   ```
3. Create a new branch for your feature:
   ```bash
   $ git checkout -b feature-branch
   ```
4. Make your changes and test thoroughly.
5. Submit a pull request with a detailed description of your changes.

### Reporting Issues
If you encounter any issues, please report them in the GitHub **Issues** section with detailed descriptions, logs, and screenshots if applicable.

---

## License

Stainalyzer is released under the **MIT License**, which allows for modification, distribution, and private use with proper attribution.

---

## Contact

For inquiries, support, or collaboration opportunities, reach out via:

- GitHub Issues: [https://github.com/yourusername/stainalyzer/issues](https://github.com/yourusername/stainalyzer/issues)
- Email: [your.email@example.com](mailto:your.email@example.com)

Thank you for using **Stainalyzer**!

---

