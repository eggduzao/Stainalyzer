import torch
import time

# Automatically choose GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Debugging

# Matrix size: Increase this for longer runtimes on CPU
MATRIX_SIZE = 1000 

# Create two large random matrices
A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)

print("Starting matrix multiplication...")

# Measure execution time of simple matrix multiplication
start_time = time.time()
C = torch.matmul(A, B)
end_time = time.time()

print(f"Matrix multiplication completed in {end_time - start_time:.2f} seconds")

