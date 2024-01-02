import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_svd(A):
    """
    Compute Singular Value Decomposition (SVD) of matrix A using NumPy.
    Returns matrices U, S, V such that A = U @ S @ V.T.
    """
    # Compute eigenvalues and eigenvectors of A.T @ A
    eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
    
    # Sort the eigenvalues in decreasing order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    # Compute diagonal matrix of singular values from eigenvalues
    singular_values = np.sqrt(eigenvalues)
    S = np.diag(singular_values)
    
    # Compute matrix U
    U = A @ eigenvectors
    U = U / singular_values
    
    # Compute matrix V
    V = eigenvectors
    
    return U, S, V.T

def total_least_squares(data):
    # Construct data matrix
    X = np.hstack((data[:, :2], np.ones((data.shape[0], 1))))
    Y = data[:, 2].reshape((-1, 1))
    n = data.shape[0]
    
    # Construct augmented matrix
    Z = np.hstack((X, Y))
    
    # Compute SVD of Z
    U, S, Vt = compute_svd(Z)
    
    # Extract smallest singular value and corresponding right singular vector
    v = Vt[-1]
    
    # Normalize v
    v /= v[-1]
    
    # Extract coefficients of plane
    a, b, c = v[:3]
    
    return a, b, c


# Load data from files
data = np.loadtxt('pc1.csv', delimiter=',')
data2 = np.loadtxt('pc2.csv', delimiter=',')

# Fit planes using total least squares method
a1, b1, c1 = total_least_squares(data)
a2, b2, c2 = total_least_squares(data2)

# Create grid of x and y values
x_range = np.arange(min(data[:, 0]), max(data[:, 0]), 0.1)
y_range = np.arange(min(data[:, 1]), max(data[:, 1]), 0.1)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)

# Evaluate planes at each x,y point
z_mesh1 = -(a1*x_mesh + b1*y_mesh + c1)
z_mesh2 = -(a2*x_mesh + b2*y_mesh + c2)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

# Plot data and plane for pc1
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o')
ax1.plot_surface(x_mesh, y_mesh, z_mesh1, alpha=0.5, color='g')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Total Least Squares Method for pc1')

# Plot data and plane for pc2
ax2.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c='b', marker='o')
ax2.plot_surface(x_mesh, y_mesh, z_mesh2, alpha=0.5, color='g')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Total Least Squares Method for pc2')

plt.show()
