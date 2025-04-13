import numpy as np
import re

def read_function(file_path):
    """
    Read the equation from the function file.

    Parameters:
        file_path (str): Path to the function file.

    Returns:
        tuple: Lists of weights and biases extracted from the equation.
    """
    with open(file_path, 'r') as file:
        equation = file.read().strip()
    
    # Extract weights and biases using regex
    weights = [float(w) for w in re.findall(r'\* ([\-]?\d+\.?\d*)', equation)]
    biases = [float(b) for b in re.findall(r'\(VALUE\d+ \+ ([\-]?\d+\.?\d*)\)', equation)]
    return np.array(weights), np.array(biases)

def z_score_normalization(matrix):
    """
    Apply Z-Score Normalization column-wise.

    Parameters:
        matrix (np.ndarray): The matrix to normalize.

    Returns:
        np.ndarray: Z-Score normalized matrix.
    """
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (matrix - mean) / std

def process_results(file_path, weights, biases):
    """
    Process the results file, apply Z-Score Normalization, and compute totals.

    Parameters:
        file_path (str): Path to the results file.
        weights (np.ndarray): Weights for the equation.
        biases (np.ndarray): Biases for the equation.

    Returns:
        list: List of tuples with metadata, z-scored values, and totals.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    elements = []
    vectors = []

    for i in range(0, len(lines), 2):
        # Metadata (first line of the element pair)
        metadata = lines[i].strip()
        
        # Extract numerical values (second line of the element pair)
        chary = [str(v) for v in lines[i + 1].strip().split('\t')[::2]]  # Take every first value (string ones)  
        values = [float(v) for v in lines[i + 1].strip().split('\t')[1::2]]  # Take every second value (numerical ones)
        vector = np.array(values)  # Store as a NumPy array
        elements.append((metadata, chary, vector))  # Append metadata and the full vector
        vectors.append(vector)  # Add vector to the list for normalization

    # Stack all vectors for Z-Score normalization
    all_vectors = np.array(vectors)
    normalized_vectors = z_score_normalization(all_vectors)

    # Compute totals using the equation
    results = []
    for (metadata, chary, _), norm_vector in zip(elements, normalized_vectors):
        total = np.sum(norm_vector + (norm_vector * weights) + biases)
        results.append((metadata, chary, norm_vector, total))

    # Sort results by total
    results.sort(key=lambda x: x[3])  # Sort by the total value (ascending)

    return results

def write_output(file_path, results):
    """
    Write the sorted results to a file.

    Parameters:
        file_path (str): Path to the output file.
        results (list): Processed and sorted results.
    """
    with open(file_path, 'w') as file:
        for metadata, chary, norm_vector, total in results:
            combined_values = '\t'.join([f'{c}\t{v:.4f}' for c, v in zip(chary, norm_vector)])
            file.write(f"{metadata}\n")
            file.write(f"{combined_values}\t")
            file.write(f"Total\t{total:.4f}\n")

def main():
    # File paths
    function_file = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/supercolider4/output_equation.txt"
    results_file = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/supercolider4/results.txt"
    output_file = "/Users/egg/Projects/Stainalyzer/examples/input_test_2/supercolider4/sorted_results.txt"

    # Read the function
    weights, biases = read_function(function_file)

    # Process results
    sorted_results = process_results(results_file, weights, biases)

    # Write the output
    write_output(output_file, sorted_results)

if __name__ == "__main__":
    main()

