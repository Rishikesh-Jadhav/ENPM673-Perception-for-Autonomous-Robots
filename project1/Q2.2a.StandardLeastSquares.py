import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def standard_least_squares(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    A = np.stack((x, y, np.ones((len(x)), dtype=int)), axis=1)
    A_t = A.transpose()
    A_tA = A_t.dot(A)
    A_tZ = A_t.dot(z)
    x_bar = np.linalg.inv(A_tA).dot(A_tZ)
    return x_bar

# Load the data
data = np.loadtxt('pc1.csv', delimiter=',')
data2 = np.loadtxt('pc2.csv', delimiter=',')

# Fitting a plane using Standard Least Squares method for data1
x_bar = standard_least_squares(data)
x = data[:, 0]
y = data[:, 1]
plane = np.array([x_bar[0]*x + x_bar[1]*y + x_bar[2] for x, y in zip(x, y)])

# Plotting the data points and fitted plane for data1
fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(x, y, data[:,2], marker='o', color='blue')
ax.plot_trisurf(x, y, plane, alpha=0.5, color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Standard Least Square for pc1')

# Fitting a plane using Standard Least Squares method for data2
x_bar = standard_least_squares(data2)
x = data2[:, 0]
y = data2[:, 1]
plane = np.array([x_bar[0]*x + x_bar[1]*y + x_bar[2] for x, y in zip(x, y)])

# Plotting the data points and fitted plane for data2
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(x, y, data2[:,2], marker='o', color='blue')
ax2.plot_trisurf(x, y, plane, alpha=0.5, color='red')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Standard Least Square for pc2')

plt.show()
