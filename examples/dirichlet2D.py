
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_dirichlet_triangle(alpha, sample_point=None):
    """
    Draws a 2-simplex (triangle) representation of a Dirichlet distribution with given alpha parameters.
    
    Parameters:
    alpha (list): Dirichlet distribution parameters [α1, α2, α3]
    sample_point (list): Optional specific point [p1, p2, p3] to highlight on the simplex
    """
    # Generate samples from the Dirichlet distribution
    samples = np.random.dirichlet(alpha, size=500)

    # Convert to 2D simplex coordinates
    x = samples[:, 0] + 0.5 * samples[:, 1]
    y = np.sqrt(3) / 2 * samples[:, 1]

    # Triangle vertices
    triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    
    # Plot the simplex
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color='blue', alpha=0.3, s=5, label="Dirichlet Samples")
    
    # Draw triangle edges
    for i in range(3):
        plt.plot([triangle_vertices[i, 0], triangle_vertices[(i+1)%3, 0]],
                 [triangle_vertices[i, 1], triangle_vertices[(i+1)%3, 1]], 'k-', lw=2)

    # Add labels at the vertices
    plt.text(-0.05, -0.05, r'$p_1$', fontsize=12, ha='center', va='center')
    plt.text(1.05, -0.05, r'$p_2$', fontsize=12, ha='center', va='center')
    plt.text(0.5, np.sqrt(3)/2 + 0.05, r'$p_3$', fontsize=12, ha='center', va='center')

    # Highlight specific sample point if provided
    if sample_point:
        px = sample_point[0] + 0.5 * sample_point[1]
        py = np.sqrt(3) / 2 * sample_point[1]
        plt.scatter(px, py, color='red', s=100, edgecolors='black', label="Given Point (0.1, 0.3, 0.6)")

    # Formatting
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, np.sqrt(3)/2 + 0.1)
    plt.axis('off')
    plt.legend()
    plt.title("Dirichlet Distribution on a 2-Simplex (Triangle)")
    plt.show()

# Function to convert Dirichlet (p1, p2, p3, p4) to 3D Cartesian coordinates
def dirichlet_to_cartesian(p1, p2, p3, p4):
    """Converts a Dirichlet (p1, p2, p3, p4) point to Cartesian coordinates inside a tetrahedron."""
    x = 0.5 * (2 * p2 + p3) / (p1 + p2 + p3 + p4)
    y = (np.sqrt(3) / 2) * (p3 / (p1 + p2 + p3 + p4))
    z = (np.sqrt(6) / 3) * (p4 / (p1 + p2 + p3 + p4))
    return x, y, z

# Function to plot the Dirichlet tetrahedron
def plot_dirichlet_tetrahedron(alpha, sample_point=None):
    """
    Plots a Dirichlet distribution with 4 probabilities inside a tetrahedron.

    Parameters:
    alpha (list): Dirichlet distribution parameters [alpha1, alpha2, alpha3, alpha4]
    sample_point (list): Optional specific point [p1, p2, p3, p4] to highlight
    """
    # Generate samples from the Dirichlet distribution
    samples = np.random.dirichlet(alpha, size=500)
    
    # Convert to 3D Cartesian coordinates
    x, y, z = zip(*[dirichlet_to_cartesian(*s) for s in samples])
    
    # Tetrahedron vertices
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3)/2, 0], [0.5, np.sqrt(3)/6, np.sqrt(6)/3]])
    faces = [[vertices[0], vertices[1], vertices[2]],
             [vertices[0], vertices[1], vertices[3]],
             [vertices[1], vertices[2], vertices[3]],
             [vertices[2], vertices[0], vertices[3]]]
    
    # Create plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='blue', alpha=0.3, s=5, label="Dirichlet Samples")
    
    # Draw tetrahedron
    poly3d = Poly3DCollection(faces, alpha=0.1, linewidths=1, edgecolors='k')
    ax.add_collection3d(poly3d)
    
    # Highlight sample point if provided
    if sample_point:
        sx, sy, sz = dirichlet_to_cartesian(*sample_point)
        ax.scatter(sx, sy, sz, color='red', s=100, edgecolors='black', label="Given Point")
    
    # Label vertices
    ax.text(0, 0, -0.05, r'$p_1$', fontsize=12, ha='center')
    ax.text(1, 0, -0.05, r'$p_2$', fontsize=12, ha='center')
    ax.text(0.5, np.sqrt(3)/2, -0.05, r'$p_3$', fontsize=12, ha='center')
    ax.text(0.5, np.sqrt(3)/6, np.sqrt(6)/3 + 0.05, r'$p_4$', fontsize=12, ha='center')
    
    # Formatting
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_zlim([-0.1, 1.1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("Dirichlet Distribution in a Tetrahedron")
    ax.legend()
    plt.show()

# Define moderate Dirichlet alpha values
alpha = [2, 3, 4, 5]

# Given sample point (p1=0.1, p2=0.2, p3=0.3, p4=0.4)
sample_point = [0.1, 0.2, 0.3, 0.4]

# Plot the Dirichlet tetrahedron
plot_dirichlet_tetrahedron(alpha, sample_point)

# # Define alpha parameters for a moderate Dirichlet distribution
alpha = [2, 3, 5]  # Moderate alpha values (adjusts shape)

# # Given sample point p_1=0.1, p_2=0.3, p_3=0.6
sample_point = [0.1, 0.3, 0.6]

# # Draw the Dirichlet distribution triangle with the given point
draw_dirichlet_triangle(alpha, sample_point)
