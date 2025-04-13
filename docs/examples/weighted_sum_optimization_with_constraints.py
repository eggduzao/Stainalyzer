import numpy as np
from scipy.optimize import minimize
import os

# Constants
SEED = 1987
np.random.seed(SEED)

class WeightedSumOptimizer:
    """
    Optimizer for calculating feature weights to achieve a desired ordering of instances.
    """

    def __init__(self, vectors, order):
        """
        Initialize the optimizer.

        Parameters:
            vectors (np.ndarray): The feature vectors of all instances.
            order (np.ndarray): Desired order of the instances (lower values indicate higher priority).
        """
        self.vectors = vectors
        self.order = order
        self.num_features = vectors.shape[1]
        self.weights = None
        self.biases = None

    def _objective_function(self, params):
        """
        Objective function to minimize the violation of the desired order.

        Parameters:
            params (np.ndarray): Current weights and biases for optimization.

        Returns:
            float: The sum of squared violations of the order constraints.
        """
        weights = params[:self.num_features]
        biases = params[self.num_features:]

        # Adjust contributions
        weight_penalty = 10  # Emphasize weights
        bias_penalty = 1  # Less emphasis on biases

        scores = np.dot(self.vectors + biases, weights)
        violations = 0

        for i in range(len(self.order) - 1):
            if self.order[i] < self.order[i + 1]:
                # Penalize violations
                violations += weight_penalty * max(0, scores[i] - scores[i + 1]) ** 2

        # Add a small penalty to encourage smaller biases
        violations += bias_penalty * np.sum(biases ** 2)

        return violations

    def _constraints(self, params):
        """
        Generate constraints to enforce the desired order.

        Parameters:
            params (np.ndarray): Current weights and biases for optimization.

        Returns:
            list: Constraint values for optimization.
        """
        weights = params[:self.num_features]
        biases = params[self.num_features:]

        # Apply biases to each feature before calculating scores
        scores = np.dot(self.vectors + biases, weights)

        # Convert self.order to indices and compute normalized differences
        ordered_indices = np.argsort(self.order)  # Relative indices for the order
        normalized_diff = (self.order[ordered_indices[:-1]] - self.order[ordered_indices[1:]])
        normalized_diff = np.abs(normalized_diff / np.max(np.abs(normalized_diff)))  # Scale to [0, 1]

        # Apply constraints A < B, with weights
        return [
            (scores[ordered_indices[i]] - scores[ordered_indices[i + 1]]) * normalized_diff[i]
            for i in range(len(ordered_indices) - 1)
        ]

    def optimize(self):
        """
        Perform the optimization to calculate weights and biases.

        Returns:
            tuple: Optimized weights and biases.
        """
        # Initial guesses for weights and biases
        initial_params = np.random.uniform(1, 10, self.num_features * 2)

        # Bounds: Weights >= 1.0, Biases unrestricted
        bounds = [(None, None)] * self.num_features + [(None, None)] * self.num_features

        # Constraints: Add constraints for the order
        constraints = {
            'type': 'ineq',
            'fun': self._constraints
        }

        def debug_callback(params):
            objective_value = self._objective_function(params)
            print(f"Iteration: Objective Value: {objective_value}, Parameters: {params}")

        result = minimize(
            self._objective_function,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'eps': 0.1,  # Minimum step size
                'ftol': 1e-2,  # Convergence tolerance
                'maxiter': 100000 # Maximum Iterations
            },
            callback=debug_callback  # Add this callback
        )

        # Success verification
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Split the result into weights and biases
        self.weights = result.x[:self.num_features]
        self.biases = result.x[self.num_features:]

        return self.weights, self.biases

    def generate_equation(self):
        """
        Generate the equation representing the weighted sum.

        Returns:
            str: A string representing the equation.
        """
        terms = []
        for i in range(self.num_features):
            terms.append(
                f"((VALUE{i + 1} + {self.biases[i]:.2f}) * {self.weights[i]:.2f})"
            )
        return " + ".join(terms)

def z_score_normalization(vectors):
    """
    Apply Z-Score Normalization to the input vectors.

    Parameters:
        vectors (np.ndarray): The feature vectors to be normalized.

    Returns:
        np.ndarray: Z-Score normalized feature vectors.
    """
    mean = np.mean(vectors, axis=0)
    std = np.std(vectors, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    normalized_vectors = (vectors - mean) / std
    return normalized_vectors

def read_data(file_path, normalize=True):
    """
    Read input data from a tab-separated file and optionally normalize it.

    Parameters:
        file_path (str): Path to the input file.
        normalize (bool): Whether to apply Z-Score Normalization.

    Returns:
        tuple: An array of feature vectors and an array of desired orders.
    """
    data = np.loadtxt(file_path, delimiter='\t')
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError("Input data must have at least two columns.")
    
    order = data[:, 0].astype(int)
    vectors = data[:, 1:]
    
    if normalize:
        vectors = z_score_normalization(vectors)
    
    return vectors, order

def write_output(file_path, equation):
    """
    Write the generated equation to a file.

    Parameters:
        file_path (str): Path to the output file.
        equation (str): The equation to write.
    """
    with open(file_path, 'w') as f:
        f.write(equation)

def main(input_file, output_file):
    """
    Main function to execute the optimization.

    Parameters:
        input_file (str): Path to the input file containing vectors and orders.
        output_file (str): Path to the output file to save the generated equation.
    """
    vectors, order = read_data(input_file)
    optimizer = WeightedSumOptimizer(vectors, order)
    weights, biases = optimizer.optimize()
    equation = optimizer.generate_equation()
    write_output(output_file, equation)
    print(f"Optimized Weights: {weights}")
    print(f"Optimized Biases: {biases}")
    print(f"Generated Equation: {equation}")

if __name__ == "__main__":

    # Example usage
    input_file = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/supercolider4/vectors.txt"
    output_file = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/supercolider4/output_equation.txt"
    main(input_file, output_file)
