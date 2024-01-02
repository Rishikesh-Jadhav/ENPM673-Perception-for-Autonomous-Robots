import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TASKS-
# 1.Compute Covariance Matrix
# 2.Using pc1  covariance matrix to compute the magnitude and direction of the surface normal

df = np.loadtxt('pc1.csv', delimiter=',')

# Calculating mean of each column and subtract from data
mean = np.mean(df, axis=0)
df -= mean

# Calculating covariance matrix using formula
covariance_matrix = np.dot(df.T, df) / (df.shape[0] - 1)

print(covariance_matrix)

# Finding eigenvectors and eigenvalues of covariance matrix
eigvals, eigvecs = np.linalg.eig(covariance_matrix)

# Finding index of smallest eigenvalue
index = np.argmin(eigvals)

# Calculating direction and magnitude of surface normal
normal = eigvecs[:, index]
magnitude = np.sqrt(eigvals[index])

print("Direction of surface normal:", normal)
print("Magnitude of surface normal:", magnitude)

# Plotting data points and surface normal (Direction of SURFACE NORMAL is colored in red)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df[:, 0], df[:, 1], df[:, 2], c='blue', alpha=0.5)
ax.plot([mean[0], mean[0]+normal[0]], [mean[1], mean[1]+normal[1]], [mean[2], mean[2]+normal[2]], c='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()