
import os
import torch

"""
# Get the number of available CPUs
num_cpus = os.cpu_count()
print(f"Number of CPUs: {num_cpus}")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
else:
    print("No GPUs available")

"""

"""
# Without Parallelisation

import numpy as np
from scipy.stats import gaussian_kde
import time

# Hypothetical pixel data (100 pixels with 10-dimensional HSV values)
pixels = np.random.rand(100, 10)  # Each pixel has 10 HSV values

# Hypothetical distributions (30 clusters, each with mean and covariance)
clusters = [
    {"mean": np.random.rand(10), "cov": np.eye(10)} for _ in range(30)
]

# Non-parallelized posterior probability calculation
def calculate_posterior(pixel, clusters):
    probabilities = []
    for cluster in clusters:
        # Generate 10,000 samples per cluster for KDE
        samples = np.random.multivariate_normal(cluster["mean"], cluster["cov"], size=10000).T
        kde = gaussian_kde(samples)
        probabilities.append(kde(pixel))
    return probabilities

# Measure time for non-parallelized computation
start_time = time.time()

# Calculate for all pixels
results = [calculate_posterior(pixel, clusters) for pixel in pixels]

end_time = time.time()
print(f"Non-parallelized execution time: {end_time - start_time:.2f} seconds")

#python test_parallelization.py
#Non-parallelized execution time: 57.29 seconds
#python test_parallelization.py
#Non-parallelized execution time: 54.14 seconds
"""

# With Parallelisation

import numpy as np
from scipy.stats import gaussian_kde
from concurrent.futures import ProcessPoolExecutor
import time

# Parallelized posterior probability calculation
def calculate_posterior_parallel(args):
    pixel, clusters = args
    probabilities = []
    for cluster in clusters:
        # Generate 10,000 samples per cluster for KDE
        samples = np.random.multivariate_normal(cluster["mean"], cluster["cov"], size=10000).T
        kde = gaussian_kde(samples)
        probabilities.append(kde(pixel))
    return probabilities


if __name__ == '__main__':

    # Hypothetical pixel data (100 pixels with 10-dimensional HSV values)
    pixels = np.random.rand(100, 10)  # Each pixel has 10 HSV values

    # Hypothetical distributions (30 clusters, each with mean and covariance)
    clusters = [
        {"mean": np.random.rand(10), "cov": np.eye(10)} for _ in range(30)
    ]

    # Prepare data to pass to the function
    args = [(pixel, clusters) for pixel in pixels]

    # Measure time for parallelized computation
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=6) as executor:  # Use 4 workers
        results = list(executor.map(calculate_posterior_parallel, args))

    end_time = time.time()
    print(f"Parallelized execution time: {end_time - start_time:.2f} seconds")

#python test_parallelization.py
#Parallelized execution time: 121.25 seconds 4 CPUs
#python test_parallelization.py
#Parallelized execution time: 118.28 seconds 6 CPUs

