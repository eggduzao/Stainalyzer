
######################################################################################################
# Contagem de Pixels
#   3752-20     913916
#   7716-20G    798591
#   8414-20F    0
#   3833-21H    0
#   5086-21F    194889
#   5829-21B    479505
#   6520-21E    815705
######################################################################################################

"""

### **Explanation of the Code**
1. **Divide the Image**:
   - The image is divided into squares of size `kernel_size x kernel_size`. For smaller edges, it adjusts the size dynamically.

2. **Save Kernel Images**:
   - Each square is saved as an image named `kernel_i_j.png`, where `i` and `j` are row and column indices.

3. **HSV Conversion**:
   - Each square is converted to the HSV color space.

4. **Generate and Save Histograms**:
   - Separate histograms for **Hue**, **Saturation**, and **Value** are calculated.
   - The histograms are saved as `histogram_i_j.png`.

5. **Output Folder**:
   - All output files are saved in a folder named `kernels`, created in the same directory as the script.

---

### **How to Run**
1. Save the script to a file (e.g., `process_kernels.py`).
2. Replace `example_image.jpg` with the path to your image.
3. Run the script:
   ```bash
   python process_kernels.py
   ```
4. Check the `kernels` folder for the generated images and histograms.

------

Using Bayesian Posterior Probabilities for your problem is a great choice, especially given the possibility of multiple reference distributions. Here’s how we can define the prior probability and the overall framework for your case:

1. The Bayesian Framework

The posterior probability ￼ is given by Bayes’ Theorem:
￼
    •   Likelihood ￼:
    •   This comes from the observed histogram (or distribution) of Hue, Saturation, and Value in the given kernel, compared to the reference distributions.
    •   Prior ￼:
    •   Encodes your prior belief about how likely certain HSV ranges are to belong to “perfect immunostained” regions.
    •   Evidence ￼:
    •   A normalizing constant. In practice, it’s often ignored during relative comparisons.

2. Setting the Prior Probability

Your prior should penalize blueish, greenish, and grayish zones (unwanted areas) and favor brownish/reddish areas (desired zones). Here’s how you can encode this:

Option 1: Uniform with Penalization

    •   Define a baseline uniform prior over all HSV values.
    •   Add a penalty function for HSV ranges corresponding to undesired colors:
    •   Hue:
    •   Penalize hues corresponding to blues and greens (e.g., ￼).
    •   Favor hues for browns and reds (e.g., ￼).
    •   Saturation:
    •   Favor high saturation (e.g., ￼).
    •   Penalize low saturation (pale/grayish regions).
    •   Value:
    •   Favor mid-range values (not too dark or too bright).

Mathematically, the prior can look like:
￼

Option 2: Data-Driven Prior

    •   Use a frequency histogram from a set of “perfect immunostained” examples as the prior.
    •   For each ￼, compute:
￼

This creates a natural prior based on real data. It inherently penalizes zones not observed in the reference distributions.

Option 3: Weighted Combination of Priors

    •   Combine a uniform penalized prior with a data-driven prior:
￼
    •   Tune ￼ (e.g., 0.5) to balance between penalization and data-based belief.

3. Defining the Likelihood

The likelihood ￼ represents how well the observed HSV histogram matches the reference distribution. Possible approaches include:
    •   Gaussian Likelihood:
    •   Assume the reference distributions are Gaussian:
￼
    •   ￼: Mean of the reference distribution for ￼.
    •   ￼: Standard deviation of the reference distribution for ￼.
    •   Histogram Likelihood:
    •   Treat the reference histogram as a discrete distribution:
￼

4. Calculating the Posterior

Combine the likelihood and prior:
    •   If you have multiple reference distributions, calculate:
￼
    •   ￼: Weight of the ￼-th reference distribution.
    •   ￼: Prior for the ￼-th reference.

5. Interpreting the Score

The posterior ￼ provides a score for how likely the HSV distribution of a kernel matches the reference distribution. Use this score to:
    •   Rank kernels by their similarity to “perfect immunostained” regions.
    •   Set a threshold to classify regions as “immunostained” or not.

Summary

    •   Use a uniform penalized prior or a data-driven prior to encode your prior belief.
    •   Define the likelihood based on Gaussian or histogram-based distributions.
    •   Combine the prior and likelihood to calculate a posterior probability for each kernel.
    •   Use the posterior as the final “similarity score” to evaluate each kernel.


"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def divide_and_analyze_image_with_posteriors(image_path, kernel_size):
    """
    Divide an image into kernels, calculate HSV histograms, and compute posterior probabilities
    for brown-red-yellow-ish and green-blue-gray-ish distributions.

    :param image_path: Path to the input image.
    :param kernel_size: Size of the kernel (square dimensions in pixels).
    """
    # Define prior probabilities
    PRIOR_BROWNISH = 0.5
    PRIOR_GREENISH = 0.3
    #PRIOR_REDDISH = 0.2

    # Define reference mean and standard deviations for each color category
    BROWNISH_PARAMS = {
        "hue": (61.72493506493507, 57.56675448901372),  # Mean = 20, Std = 10
        "saturation": (45.917337662337665, 25.483717453496073),  # Mean = 150, Std = 50
        "value": (105.44961038961038, 13.581808157797575)  # Mean = 150, Std = 50
    }
    GREENISH_PARAMS = {
        "hue": (108.63273092369478, 17.747042656852557),  # Mean = 120, Std = 40
        "saturation": (55.121004016064255, 31.295240290743983),  # Mean = 100, Std = 40
        "value": (118.38441767068274, 10.521819658116208)  # Mean = 100, Std = 40
    }

    #REDDISH_PARAMS = {
    #"hue": (10, 5),  # Example mean = 10, std = 5
    #"saturation": (200, 30),  # Example mean = 200, std = 30
    #"value": (180, 40)  # Example mean = 180, std = 40
    #}

    def gaussian_probability(x, mean, std):
        """Calculate Gaussian probability density function."""
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image could not be loaded. Check the file path.")
        return
    height, width = image.shape[:2]

    # Create a folder to save the output
    output_folder = "kernels_with_posteriors"
    os.makedirs(output_folder, exist_ok=True)

    # Process the image and divide it into squares
    for i, y in enumerate(range(0, height, kernel_size)):
        for j, x in enumerate(range(0, width, kernel_size)):
            # Calculate the size of the current square
            square_height = kernel_size if y + kernel_size <= height else height - y
            square_width = kernel_size if x + kernel_size <= width else width - x

            # Extract the square from the image
            square = image[y:y + square_height, x:x + square_width]

            # Convert the square to HSV
            hsv_square = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv_square)

            # Calculate mean Hue, Saturation, and Value for the kernel
            mean_hue = np.mean(hue)
            mean_saturation = np.mean(saturation)
            mean_value = np.mean(value)

            # Calculate likelihoods for both distributions
            likelihood_brownish = (
                gaussian_probability(mean_hue, *BROWNISH_PARAMS["hue"]) *
                gaussian_probability(mean_saturation, *BROWNISH_PARAMS["saturation"]) *
                gaussian_probability(mean_value, *BROWNISH_PARAMS["value"])
            )
            likelihood_greenish = (
                gaussian_probability(mean_hue, *GREENISH_PARAMS["hue"]) *
                gaussian_probability(mean_saturation, *GREENISH_PARAMS["saturation"]) *
                gaussian_probability(mean_value, *GREENISH_PARAMS["value"])
            )
            
            #likelihood_reddish = (
            #    gaussian_probability(mean_hue, *REDDISH_PARAMS["hue"]) *
            #    gaussian_probability(mean_saturation, *REDDISH_PARAMS["saturation"]) *
            #    gaussian_probability(mean_value, *REDDISH_PARAMS["value"])
            #)

            # Calculate posterior probabilities using Bayes' theorem
            posterior_brownish = (
                likelihood_brownish * PRIOR_BROWNISH
            ) / (likelihood_brownish * PRIOR_BROWNISH + likelihood_greenish * PRIOR_GREENISH)
            posterior_greenish = (
                likelihood_greenish * PRIOR_GREENISH
            ) / (likelihood_brownish * PRIOR_BROWNISH + likelihood_greenish * PRIOR_GREENISH)

            #posterior_reddish = (
            #    likelihood_reddish * PRIOR_REDDISH
            #) / (
            #    likelihood_brownish * PRIOR_BROWNISH +
            #    likelihood_greenish * PRIOR_GREENISH +
            #    likelihood_reddish * PRIOR_REDDISH
            #)

            # Save results and visualizations
            kernel_filename = os.path.join(output_folder, f"kernel_{i}_{j}.png")
            cv2.imwrite(kernel_filename, square)

            plt.figure(figsize=(8, 5))
            plt.bar(["Brownish", "Greenish"], [posterior_brownish, posterior_greenish], color=["brown", "green"])
            plt.title(f"Posteriors for kernel_{i}_{j}")
            plt.ylabel("Posterior Probability")
            plt.ylim(0, 1)
            posterior_filename = os.path.join(output_folder, f"posterior_{i}_{j}.png")
            plt.savefig(posterior_filename)
            plt.close()

    print(f"Kernel images and posteriors saved in the folder: {output_folder}")

# Example usage
image_path = "./3753-21C ACE-2 40X(2).jpg"  # Replace with your image path
kernel_size = 10  # Define the kernel size in pixels
divide_and_analyze_image_with_posteriors(image_path, kernel_size)


