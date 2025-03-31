
"""
Tester
------

Placeholder.
"""

############################################################################################################
### Import
############################################################################################################

import io
import os
import cv2
import tempfile
import numpy as np
import seaborn as sns
from pathlib import Path
from time import perf_counter
from PIL import Image, ImageCms
import matplotlib.pyplot as plt

from .loader import MetricTables

############################################################################################################
### Classes
############################################################################################################

class Tester:
    """
    A class to manage the evaluation and testing process for trained distribution models.

    The Tester class is responsible for:
    - Loading and preprocessing test data.
    - Evaluating the performance of trained models on test datasets.
    - Comparing trained models using various metrics.
    - Generating detailed evaluation reports.
    - Logging progress and results for transparency.

    Attributes:
        model (object): The trained model to be tested (e.g., DABDistribution, MultivariateLABDistribution).
        test_data (list or numpy.ndarray): The dataset used for testing.
        metrics (dict): Dictionary of metrics to evaluate model performance.
        logger (object, optional): Logger for tracking testing progress and results.
    """

    def __init__(self,
                 model_path : Path,
                 model : Any = None,
                 test_data : pd.DataFrame = None,
                 metrics : pd.DataFrame = None,
                 logger : pd.DataFrame = None):
        """
        Initialize the Tester class.

        Parameters:
            model (object, optional): The trained model to be tested. Default is None.
            test_data (list or numpy.ndarray, optional): Dataset for testing. Default is None.
            metrics (dict, optional): Evaluation metrics to use. Default is None.
            logger (object, optional): Logger for tracking progress. Default is None.
        """
        self.model_path = model_path if model_path.exist() else None
        self.model = model  # The trained model to be tested
        self.test_data = test_data  # The dataset to evaluate the model on
        self.metrics = metrics if metrics else {}  # Default to an empty dictionary if no metrics are provided
        self.logger = logger  # Logger instance for progress tracking

        # TODO - Validate and preprocess the test_data if required
        if self.test_data is not None:
            self.preprocess_test_data()

    def load_test_data(self, data_path):
        """
        Load test data from a specified path.

        Parameters:
            data_path (str): Path to the test data.

        Returns:
            None

        Raises:
            ValueError: If the specified path is invalid or the data cannot be loaded.
        """
        # TODO - Validate if the provided path exists
        if not os.path.exists(data_path):
            raise ValueError(f"Specified path does not exist: {data_path}")

        # TODO - Load all image files from the provided path
        self.test_data = []
        valid_extensions = (".png", ".jpg", ".jpeg", ".tiff")
        for file_name in os.listdir(data_path):
            if file_name.lower().endswith(valid_extensions):
                file_path = os.path.join(data_path, file_name)
                try:
                    # Load the image (assuming OpenCV is used)
                    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    if image is not None:
                        self.test_data.append((file_name, image))
                    else:
                        print(f"[WARNING]: Could not load image: {file_path}")
                except Exception as e:
                    print(f"[ERROR]: Failed to load {file_path}. Reason: {e}")

        # Validate loaded data
        if not self.test_data:
            raise ValueError(f"No valid images found in the specified path: {data_path}")

        print(f"[TEST DATA LOADED]: {len(self.test_data)} images loaded from {data_path}.")

    def preprocess_test_data(self, **kwargs):
        """
        Preprocess the test data to prepare it for evaluation.

        Parameters:
            **kwargs: Additional parameters for preprocessing, such as SLIC parameters, CLAHE settings, or K-means options.

        Returns:
            None

        Raises:
            ValueError: If no test data is loaded prior to preprocessing.
        """
        # Check if test data exists
        if not hasattr(self, 'test_data') or not self.test_data:
            raise ValueError("Test data is not loaded. Please load test data before preprocessing.")

        # Initialize the ImagePreprocessor
        # TODO - Validate if the ImagePreprocessor class is initialized correctly with parameters
        preprocessor = ImagePreprocessor(**kwargs)

        # Process each image in the test dataset
        self.preprocessed_test_data = []
        for file_name, image in self.test_data:
            try:
                # Step 1: Remove annotations if applicable
                # TODO - Ensure the annotation removal step is implemented in the ImagePreprocessor
                preprocessor.set_image(image)
                preprocessor.remove_annotation()

                # Step 2: Apply CLAHE normalization
                # TODO - Ensure CLAHE is implemented and correctly parametrized in ImagePreprocessor
                preprocessor.apply_clahe()

                # Step 3: Apply SLIC segmentation (specific to test images)
                # TODO - Ensure SLIC is implemented and takes relevant parameters
                slic_segments = preprocessor.apply_slic()

                # Step 4: Apply K-means clustering (specific to test images)
                # TODO - Ensure K-means clustering is implemented in the ImagePreprocessor
                quantized_image, centroids, pixel_counts, cluster_labels = preprocessor.apply_kmeans()

                # Store processed image details
                self.preprocessed_test_data.append({
                    'file_name': file_name,
                    'processed_image': preprocessor.processed_image,
                    'slic_segments': slic_segments,
                    'quantized_image': quantized_image,
                    'centroids': centroids,
                    'pixel_counts': pixel_counts,
                    'cluster_labels': cluster_labels
                })
            except Exception as e:
                print(f"[ERROR]: Preprocessing failed for {file_name}. Reason: {e}")

        print(f"[PREPROCESSING COMPLETE]: {len(self.preprocessed_test_data)} test images preprocessed.")

    def evaluate(self, **kwargs):
        """
        Evaluate the model on the test data using specified metrics.

        Parameters:
            **kwargs: Additional parameters for the evaluation process, such as metrics to compute
                      and thresholds for classification.

        Returns:
            dict: A dictionary of metric names and their corresponding values.

        Raises:
            ValueError: If test data or a fitted model is not available.
        """
        # Check if test data exists
        if not hasattr(self, 'preprocessed_test_data') or not self.preprocessed_test_data:
            raise ValueError("Preprocessed test data is not available. Please preprocess the test data first.")

        # Check if the model is fitted
        if not self.model or not self.model.is_fitted:
            raise ValueError("A fitted model is required for evaluation. Please train or load a model first.")

        # Metrics dictionary to store evaluation results
        metrics = {}

        # Define evaluation parameters
        threshold = kwargs.get("threshold", 0.5)  # Default threshold
        metric_list = kwargs.get("metrics", ["accuracy", "precision", "recall", "f1_score"])

        # Iterate over each preprocessed test image
        for test_image in self.preprocessed_test_data:
            try:
                file_name = test_image["file_name"]
                quantized_image = test_image["quantized_image"]
                cluster_labels = test_image["cluster_labels"]

                # Get the centroids for each cluster
                centroids = test_image["centroids"]

                # Evaluate each cluster against the model
                for cluster_id, centroid in enumerate(centroids):
                    # Evaluate the centroid using the model
                    # TODO - Verify `evaluate` method in the model (MultivariateLABDistribution or DABDistribution)
                    likelihood = self.model.evaluate(centroid[np.newaxis, :])

                    # Compare the likelihood with the threshold
                    is_positive = likelihood > threshold

                    # Log or accumulate metrics
                    # TODO - Implement metric calculations such as accuracy, precision, recall
                    # Example: Increment true positives, false positives, etc.

            except Exception as e:
                print(f"[ERROR]: Evaluation failed for {file_name}. Reason: {e}")

        # Calculate final metrics
        metrics["accuracy"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1_score"] = 0.0

        print(f"[EVALUATION COMPLETE]: Metrics calculated: {metrics}")
        return metrics

    def execute(self):
        """
        Placeholder

        Returns:
            Placeholder: Placeholder.

        Raises:
            Placeholder: Placeholder.
        """
        
        # Calculate "Cell" tables
        cell_path = self.model_path / "1_cell"
        cell_path.mkdir(parents=True, exist_ok=True)
        metric_tables = MetricTables()
        metric_tables.create_cell_tables()



    def compare(self, other_model, metric='Wasserstein', **kwargs):
        """
        Compare the performance of the current model with another model.

        This method calculates the distance or similarity score between the distributions
        of the current model and another model using the specified metric.

        Parameters:
            other_model (object): Another model to compare against. It must support the `distance_to` method.
            metric (str): Metric to use for comparison (e.g., 'Wasserstein', 'KL-divergence').
                          Default is 'Wasserstein'.
            **kwargs: Additional parameters for the comparison process.

        Returns:
            float: The calculated distance or similarity score.

        Raises:
            ValueError: If the models are not fitted or the comparison metric is unsupported.
        """
        # Check if the current model is fitted
        if not self.model or not self.model.is_fitted:
            raise ValueError("The current model is not fitted. Please train or load the model before comparison.")

        # Check if the other model is valid and fitted
        if not other_model or not hasattr(other_model, 'is_fitted') or not other_model.is_fitted:
            raise ValueError("The other model is not fitted or invalid. Please provide a valid fitted model for comparison.")

        # Verify if the other model supports the `distance_to` method
        if not hasattr(self.model, 'distance_to') or not callable(getattr(self.model, 'distance_to')):
            raise ValueError("The current model does not support the `distance_to` method required for comparison.")

        if not hasattr(other_model, 'distance_to') or not callable(getattr(other_model, 'distance_to')):
            raise ValueError("The other model does not support the `distance_to` method required for comparison.")

        # Extract weights from kwargs if provided
        weights = kwargs.get("weights", None)

        # Calculate the distance or similarity score
        try:
            distance = self.model.distance_to(other_model, metric=metric, weights=weights)
        except Exception as e:
            raise RuntimeError(f"An error occurred during the comparison: {e}")

        # Log the comparison result
        print(f"[COMPARISON COMPLETE]: Metric used: {metric}, Distance/Score: {distance:.4f}")

        return distance

    def generate_report(self, output_file=None):
        """
        Generate a report summarizing the evaluation results.

        Parameters:
            output_file (str, optional): Path to save the report. If None, the report is printed to the console.

        Returns:
            None

        Notes:
            - The report summarizes the evaluation metrics, comparisons, and key insights.
            - The structure and content of the report depend on previously computed results.
        """
        # TODO - Collect evaluation results and metrics
        evaluation_results = {}  # Placeholder for the actual evaluation results
        report_lines = ["Evaluation Report for Tester", "-" * 40]

        # Add evaluation results to the report
        for metric, value in evaluation_results.items():
            report_lines.append(f"{metric}: {value:.4f}")

        # Check if additional comparisons were performed
        # TODO - Collect comparison results if available
        comparison_results = {}  # Placeholder for comparison results
        if comparison_results:
            report_lines.append("\nComparison Results:")
            for metric, value in comparison_results.items():
                report_lines.append(f"{metric}: {value:.4f}")

        # Generate the report as a string
        report = "\n".join(report_lines)

        if output_file:
            # Save the report to the specified file
            try:
                with open(output_file, "w") as file:
                    file.write(report)
                print(f"[REPORT SAVED]: The report has been saved to {output_file}")
            except Exception as e:
                raise RuntimeError(f"Failed to save the report: {e}")
        else:
            # Print the report to the console
            print(report)

    def log_results(self, message):
        """
        Log a message about the testing progress or results.

        Parameters:
            message (str): The message to log.

        Returns:
            None

        Notes:
            - This method uses the logger if available; otherwise, it prints the message to the console.
        """
        if self.logger:
            self.logger.info(message)
        else:
            print(f"[LOG]: {message}")

    def _generate(self):

        pass
        # numpy.random

        # Distribution    Function    Parameters
        # Uniform numpy.random.uniform(low, high, size=N) low (min), high (max)
        # Normal (Gaussian)   numpy.random.normal(loc, scale, size=N) loc (mean), scale (std dev)
        # Log-Normal  numpy.random.lognormal(mean, sigma, size=N) mean, sigma
        # Exponential numpy.random.exponential(scale, size=N) scale (1/λ)
        # Gamma   numpy.random.gamma(shape, scale, size=N)    shape, scale
        # Beta    numpy.random.beta(a, b, size=N) a, b
        # Chi-Square  numpy.random.chisquare(df, size=N)  df (degrees of freedom)
        # Dirichlet   numpy.random.dirichlet(alpha, size=N)   alpha (parameter vector)
        # Poisson numpy.random.poisson(lam, size=N)   lam (λ, mean number of events)
        # Binomial    numpy.random.binomial(n, p, size=N) n (trials), p (success prob)
        # Geometric   numpy.random.geometric(p, size=N)   p (success probability)

        # scipy.stats

        # Distribution    Function    Parameters
        # Uniform scipy.stats.uniform.rvs(loc, scale, size=N) loc, scale
        # Normal (Gaussian)   scipy.stats.norm.rvs(loc, scale, size=N)    loc, scale
        # Log-Normal  scipy.stats.lognorm.rvs(sigma, loc, scale, size=N)  sigma, loc, scale
        # Exponential scipy.stats.expon.rvs(scale, size=N)    scale (1/λ)
        # Gamma   scipy.stats.gamma.rvs(shape, scale, size=N) shape, scale
        # Beta    scipy.stats.beta.rvs(a, b, size=N)  a, b
        # Chi-Square  scipy.stats.chi2.rvs(df, size=N)    df (degrees of freedom)
        # Dirichlet   scipy.stats.dirichlet.rvs(alpha, size=N)    alpha (parameter vector)
        # Poisson scipy.stats.poisson.rvs(mu, size=N) mu (mean λ)
        # Binomial    scipy.stats.binom.rvs(n, p, size=N) n, p
        # Negative Binomial   scipy.stats.nbinom.rvs(n, p, size=N)    n, p
        # Geometric   scipy.stats.geom.rvs(p, size=N) p
        # Student’s t scipy.stats.t.rvs(df, size=N)   df
        # Weibull scipy.stats.weibull_min.rvs(c, size=N)  c (shape parameter)
        # Cauchy  scipy.stats.cauchy.rvs(loc, scale, size=N)  loc, scale

    def __repr__(self):
        """
        Representation of the Tester class.

        Returns:
            str: Summary of the current state of the Tester object.
        """
        return (
            f"Tester("
            f"Model: {self.model.__class__.__name__ if self.model else 'None'}, "
            f"Test Data: {'Loaded' if self.test_data else 'None'}, "
            f"Logger: {'Enabled' if self.logger else 'Disabled'}"
            f")"
        )


if __name__ == "__main__":

    results_path = Path("/Users/egg/Projects/Stainalyzer/data/results/")
    tester = Tester(results_path)
    tester.execute()

   