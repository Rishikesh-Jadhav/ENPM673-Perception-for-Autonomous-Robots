# References : 
# https://leohart.wordpress.com/2010/07/23/rq-decomposition-from-qr-decomposition/
# https://medium.com/analytics-vidhya/camera-calibration-with-opencv-f324679c6eb7
# https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/agartia3/index.html

# This script performs camera calibration using 3D-2D correspondences and outputs the camera calibration matrix and reprojection error.

import numpy as np

# Given 3D points in the world coordinate system
world_pts = [[0,0,0,1],[0,3,0,1],[0,7,0,1],[0,11,0,1],[7,1,0,1],[0,11,7,1],[7,9,0,1],[0,1,7,1]]
world_pts = np.array(world_pts)

# Given 2D points in the image coordinate system
image_pts = [[757,213,1],[758,415,1],[758,686,1],[759,966,1],[1190,172,1],[329,1041,1],[1204,850,1],[340,159,1]]
image_pts = np.array(image_pts)

# Function to calculate the A matrix used for the camera calibration
def calculate_A(world_pts , image_pts):
    """
    Given 3D-2D correspondences, calculate the A matrix used for the camera calibration.
    """
    A = [] #2nx12 matrix

    for i in range(8):
        X_w = world_pts[i]
        u, v, w = image_pts[i]

        A_row1 = np.array([0, 0, 0, 0, -w*X_w[0], -w*X_w[1], -w*X_w[2], -w*X_w[3], v*X_w[0], v*X_w[1], v*X_w[2], v*X_w[3]])
        A_row2 = np.array([w*X_w[0], w*X_w[1], w*X_w[2], w*X_w[3], 0, 0, 0, 0, -u*X_w[0], -u*X_w[1], -u*X_w[2], -u*X_w[3]])
        A.append(A_row1)
        A.append(A_row2)

    return A

# Function to calculate the Projection matrix (3x4)
def calculate_P(A):

    """
    Given the A matrix, calculate the Projection matrix (3x4).
    """
    U, Sigma, V = np.linalg.svd(A)  
    P = np.reshape(V[-1, :], (3, 4)) #reshaping last row into 3x4 matrix
    Lambda = P[-1,-1]
    P = P/Lambda

    return P


# Function to calculate the M matrix used for RQ factorization
def calculate_M(P):
    """
    Given the Projection matrix, calculate the M matrix used for RQ factorization.
    """
    u, d, v = np.linalg.svd(P)
    C = v[-1, :]
    C = C/C[-1]
    C = np.reshape(C, (4, 1))
    I = np.identity(3)
    C = np.concatenate((I, -1*C[:-1]), axis=1)
    C_inv = np.linalg.pinv(C)
    M = np.matmul(P, C_inv)

    return M #3x3

# Function to perform RQ factorization on the M matrix 
def rqFact(M):
    """
    Given the M matrix, perform RQ factorization to obtain the rotation matrices and calibration matrix.
    """
    c = (M[2,2]/((M[2,1])**2 + (M[2,2])**2)**(0.5))
    s = -(M[2,1]/((M[2,1])**2 + (M[2,2])**2)**(0.5))
    r_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    M = np.matmul(M, r_x)
    
    c = (M[2,2]/((M[2,0])**2 + (M[2,2])**2)**(0.5))
    s = (M[2,0]/((M[2,0])**2 + (M[2,2])**2)**(0.5))
    r_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    M = np.matmul(M, r_y)
    
    c = (M[1,1]/((M[1,0])**2 + (M[1,1])**2)**(0.5))
    s = -(M[1,0]/((M[1,0])**2 + (M[1,1])**2)**(0.5))
    r_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    K = np.matmul(M, r_z)
   
    return r_x , r_y , r_z ,K #rotation matrices and calibration matrix


A = calculate_A(world_pts,image_pts)
A = np.array(A)

# Projection Matrix
P = calculate_P(A)
u, d, v = np.linalg.svd(P)

# translation vector
translation = -np.array(v.T[:,-1])

M = calculate_M(P)
rx , ry , rz , K = rqFact(M)

# Rotation matrix
R = rz @ ry @ rx

K_i = []
for i in K:
    for j in i:
        #condition to check if value is very low setting it to zero and to avoid printing very small floating-point values. 
        if abs(j) < 0.0001:
            j = 0
        K_i.append(j)
K_i = np.array(K_i)

# K matrix
K = K_i.reshape(3,3).astype(np.float32)

# calculatiing the reprojection error for each point
errors = []
for i in range(8):
    X = np.array([world_pts[i]]).T
    x = np.array([image_pts[i]]).T
    x_reproj = P @ X
    x_reproj /= x_reproj[-1]
    error = np.linalg.norm(x - x_reproj)
    errors.append(error)

print('Projection matrix : \n',P)
print('Camera calibration matrix (K) : \n',K)
print('Rotation Matrix : \n',R)
print('Translation vector : \n',translation)
print("Reprojection errors:", errors)


