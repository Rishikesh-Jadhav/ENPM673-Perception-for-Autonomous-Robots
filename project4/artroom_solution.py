# references:-
# https://cmsc733.github.io/2022/proj/p3/#fundmatrix
# https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect16.pdf
# https://medium.com/analytics-vidhya/camera-calibration-with-opencv-f324679c6eb7
# https://learnopencv.com/introduction-to-epipolar-geometry-and-stereo-vision/

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
import math

# Given K1 and K2, baseline 
K1 = np.array([[1733.74, 0, 792.27],
              [0, 1733.74, 541.89],
              [0, 0, 1]])
K2 = np.array([[1733.74, 0, 792.27],
              [0, 1733.74, 541.89],
              [0, 0, 1]])

baseline=177.288


def CalculateFundamentalMatrix(matches):
    """
    Computes A matrix from the image points found by the feature extractor.

    """
    points1 = matches[:, 0:2]
    points2 = matches[:, 2:4]

    A = np.zeros((len(points1), 9))
    for i in range(len(points1)):
        x1, y1 = points1[i][0], points1[i][1]
        x2, y2 = points2[i][0], points2[i][1]
        A[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    
    U, S, VT = np.linalg.svd(A, full_matrices=True)
    F = VT.T[:, -1]
    F = F.reshape(3,3)

    u, s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2,2] = 0
    F = np.dot(u, np.dot(s, vt))

    return F


def FundamentalMatrixError(feature, F): 
    """
    Computes the error between the points in the two images defined by the feature set and the fundamental matrix.

    """
    point1, point2 = feature[0:2], feature[2:4]
    point1_temp = np.array([point1[0], point1[1], 1]).T
    point2_temp = np.array([point2[0], point2[1], 1])

    error = np.dot(point1_temp, np.dot(F, point2_temp))
    error = np.abs(error)
   
    return error


def ComputeBestFMatrixRansac(features):
    """
    Computes the Best F matrix from the image points in the features matrix found by the feature extractor.

    """
    n_iterations = 1200
    error_thresh = 0.03
    inliers_thresh = 0
    chosen_indices = []
    Best_F = 0

    for i in range(n_iterations):
        indices = []
        n_rows = features.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        features8 = features[random_indices, :] 
        f = CalculateFundamentalMatrix(features8)
        for j in range(n_rows):
            feature = features[j]
            error = FundamentalMatrixError(feature, f)
            if error < error_thresh:
                indices.append(j)

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            chosen_indices = indices
            Best_F = f

    filtered_features = features[chosen_indices, :]
    return Best_F, filtered_features

def ComputeEssentialMatrix(K1, K2, F):
    """
    Computes Essential matrix from the K matrices of the cameras and the best_F matrix.

    """
    E = K2.T.dot(F).dot(K1)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E = np.dot(U,np.dot(np.diag(s),V))
    return E


def ExtractRC(E):
    """
    Computes the camera center vectors C and possible rotation matrices R 
    from the Essential matrix E using Singular Value Decomposition.

    """

    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    C1 = U[:, 2]
    C2 = -U[:, 2]
    C3 = U[:, 2]
    C4 = -U[:, 2]
    R1 = np.dot(U, np.dot(W, V_T))
    R2 = np.dot(U, np.dot(W, V_T))
    R3 = np.dot(U, np.dot(W.T, V_T))
    R4 = np.dot(U, np.dot(W.T, V_T))

    R = [R1, R2, R3, R4]
    C = [C1, C2, C3, C4]

    for i in range(4):
        if np.linalg.det(R[i]) < 0:
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C



def TriangulatePoints(K1, K2, matched_pairs, R2, C2):
    """
    Computes 3D points from a set of corresponding 2D points in two images, using the triangulation method.
    """
    pts3D = []
    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    I = np.identity(3)
    P1 = np.dot(K1, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))

    for i in range(len(C2)):
        x1 = matched_pairs[:,0:2].T
        x2 = matched_pairs[:,2:4].T

        P2 = np.dot(K2, np.dot(R2[i], np.hstack((I, -C2[i].reshape(3,1)))))

        X = cv2.triangulatePoints(P1, P2, x1, x2)
        pts3D.append(X)

    return pts3D

def GetPositiveZCount(pts3D, R, C):
    """
    Computes and returns the count of points that have positive Z coordinates in the camera's coordinate system.
    
    """
    I = np.identity(3)
    P = np.dot(R, np.hstack((I, -C.reshape(3,1))))
    P = np.vstack((P, np.array([0,0,0,1]).reshape(1,4)))
    n_positive_z = 0
    for i in range(pts3D.shape[1]):
        X = pts3D[:,i]
        X = X.reshape(4,1)
        Xc = np.dot(P, X)
        Xc = Xc / Xc[3]
        z = Xc[2]
        if z > 0:
            n_positive_z += 1

    return n_positive_z

def GetX(line, y):
    """
    Given a line equation in the form ax + by + c = 0, computing the x value for a given y value.
   
     """
    return -(line[1]*y + line[2])/line[0]

def GetEpiLines(set1, set2, F, img1, img2, rectified = False):
    """
    Computes the epipolar lines corresponding to the matched pairs of points in two images.

    """
    lines1, lines2 = [], []
    img_epi1 = img1.copy()
    img_epi2 = img2.copy()

    for i in range(set1.shape[0]):
        x1 = np.array([set1[i,0], set1[i,1], 1]).reshape(3,1)
        x2 = np.array([set2[i,0], set2[i,1], 1]).reshape(3,1)

        line2 = np.dot(F, x1)
        lines2.append(line2)

        line1 = np.dot(F.T, x2)
        lines1.append(line1)
    
        if not rectified:
            y2_min = 0
            y2_max = img2.shape[0]
            x2_min = GetX(line2, y2_min)
            x2_max = GetX(line2, y2_max)
            y1_min = 0
            y1_max = img1.shape[0]
            x1_min = GetX(line1, y1_min)
            x1_max = GetX(line1, y1_max)
        else:
            x2_min = 0
            x2_max = img2.shape[1] - 1
            y2_min = -line2[2]/line2[1]
            y2_max = -line2[2]/line2[1]
            x1_min = 0
            x1_max = img1.shape[1] -1
            y1_min = -line1[2]/line1[1]
            y1_max = -line1[2]/line1[1]



        cv2.circle(img_epi2, (int(set2[i,0]),int(set2[i,1])), 10, (0,0,255), -1)
        img_epi2 = cv2.line(img_epi2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 0, int(i*2.55)), 2)
    

        cv2.circle(img_epi1, (int(set1[i,0]),int(set1[i,1])), 10, (0,0,255), -1)
        img_epi1 = cv2.line(img_epi1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 0, int(i*2.55)), 2)

    concat = np.concatenate((img1, img2), axis = 1)
    concat = cv2.resize(concat, (1920, 1080))
    cv2.imshow("Epipolar lines", concat)
    cv2.imwrite("artroom_epipolar_lines.png", concat)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    return lines1, lines2


# Path to the folder containing images
path = 'artroom'

orb = cv2.ORB_create()

# Reading the images
img1 = cv2.imread(path + '/im0.png')
img2 = cv2.imread(path + '/im1.png')

# grayscaling the images
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# extracting the keypoints and descripters
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# creating a brute force matcher object
bf = cv2.BFMatcher()
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x :x.distance)
chosen_matches = matches[0:100]

# Features to arrays conversion
matched_pairs = []
for i, m1 in enumerate(chosen_matches):
    pt1 = kp1[m1.queryIdx].pt
    pt2 = kp2[m1.trainIdx].pt
    matched_pairs.append([pt1[0], pt1[1], pt2[0], pt2[1]])
matched_pairs = np.array(matched_pairs).reshape(-1, 4)

# Displaying the image with the matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, chosen_matches, img1, flags=2)
cv2.imshow("image", img3)
cv2.waitKey()
cv2.destroyAllWindows()

Best_F, matched_pairs_inliers = ComputeBestFMatrixRansac(matched_pairs)
E = ComputeEssentialMatrix(K1,K2,Best_F)
R, C = ExtractRC(E)
print("Best Fundamental matrix",Best_F)
print("Essential matrix",E)
print("Rotation matrix",R)
print("translation vector",C)

pts3D_4 = TriangulatePoints(K1, K2, matched_pairs_inliers, R, C)

z_count1 = []
z_count2 = []

R1 = np.identity(3)
C1 = np.zeros((3,1))
for i in range(len(pts3D_4)):
    pts3D = pts3D_4[i]
    pts3D = pts3D/pts3D[3, :]
    x = pts3D[0,:]
    y = pts3D[1, :]
    z = pts3D[2, :]    

    z_count2.append(GetPositiveZCount(pts3D, R[i], C[i]))
    z_count1.append(GetPositiveZCount(pts3D, R1, C1))

z_count1 = np.array(z_count1)
z_count2 = np.array(z_count2)

count_thresh = int(pts3D_4[0].shape[1] / 2)
idx = np.intersect1d(np.where(z_count1 > count_thresh), np.where(z_count2 > count_thresh))
R2_ = R[idx[0]]
C2_ = C[idx[0]]
X_ = pts3D_4[idx[0]]
X_ = X_/X_[3,:]

print("Estimated R :", R2_)
print("Estimated C :", C2_)

# Plotting epilines
set1, set2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]
lines1, lines2 = GetEpiLines(set1, set2, Best_F, img1, img2, False)

# Image rectification
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(set1), np.float32(set2), Best_F, imgSize=(w1, h1))


img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

set1_rectified = cv2.perspectiveTransform(set1.reshape(-1, 1, 2), H1).reshape(-1,2)
set2_rectified = cv2.perspectiveTransform(set2.reshape(-1, 1, 2), H2).reshape(-1,2)

img1_rectified_draw = img1_rectified.copy()
img2_rectified_draw = img2_rectified.copy()

for i in range(set1_rectified.shape[0]):
    cv2.circle(img1_rectified_draw, (int(set1_rectified[i,0]),int(set1_rectified[i,1])), 10, (0,0,255), -1)
    cv2.circle(img2_rectified_draw, (int(set2_rectified[i,0]),int(set2_rectified[i,1])), 10, (0,0,255), -1)

cv2.imshow("img1_rectified", img1_rectified_draw)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("img2_rectified", img2_rectified_draw)
cv2.waitKey()
cv2.destroyAllWindows()

H2_T_inv =  np.linalg.inv(H2.T)
H1_inv = np.linalg.inv(H1)
F_rectified = np.dot(H2_T_inv, np.dot(Best_F, H1_inv))

lines1_rectified, lines2_recrified = GetEpiLines(set1_rectified, set2_rectified, F_rectified, img1_rectified, img2_rectified, True)


# FINDING CORRESPONDENCE
# Resizing the rectified images to reduce computational complexity
img1_rectified_reshaped = cv2.resize(img1_rectified, (int(img1_rectified.shape[1] / 4), int(img1_rectified.shape[0] / 4)))
img2_rectified_reshaped = cv2.resize(img2_rectified, (int(img2_rectified.shape[1] / 4), int(img2_rectified.shape[0] / 4)))

img1_rectified_reshaped = cv2.cvtColor(img1_rectified_reshaped, cv2.COLOR_BGR2GRAY)
img2_rectified_reshaped = cv2.cvtColor(img2_rectified_reshaped, cv2.COLOR_BGR2GRAY)

window = 13

# Creating arrays for the left and right images
left_array, right_array = img1_rectified_reshaped, img2_rectified_reshaped
left_array = left_array.astype(int)
right_array = right_array.astype(int)
h, w = left_array.shape

# COMPUTING DEPTH IMAGE
# Creating an array for the disparity map
disparity_map = np.zeros((h, w))

block_left_array_2D = []
block_right_array_2D = []
# Calculating the new width of the image
x_new = w - (2 * window)


# Creating blocks of the left and right images and adding them to the 2D arrays
for y in range(window, h-window):
    block_left_array = []
    block_right_array = []
    for x in range(window, w-window):
        block_left = left_array[y:y + window, x:x + window]
        block_left_array.append(block_left.flatten())

        block_right = right_array[y:y + window, x:x + window]
        block_right_array.append(block_right.flatten())

    block_left_array = np.array(block_left_array)
    block_left_array = np.repeat(block_left_array[:, :, np.newaxis], x_new, axis=2)

    block_right_array = np.array(block_right_array)
    block_right_array = np.repeat(block_right_array[:, :, np.newaxis], x_new, axis=2)
    block_right_array = block_right_array.T

    abs_diff = np.abs(block_left_array - block_right_array)
    sum_abs_diff = np.sum(abs_diff, axis=1)
    idx = np.argmin(sum_abs_diff, axis=0)
    disparity = np.abs(idx - np.linspace(0, x_new, x_new, dtype=int)).reshape(1, x_new)
    disparity_map[y, 0:x_new] = disparity

# Converting the disparity map to an integer array and scaling it
disparity_map_int = np.uint8(disparity_map * 255 / np.max(disparity_map))
plt.imshow(disparity_map_int, cmap='hot', interpolation='nearest')
plt.savefig('artroom_depth_image.png')
plt.show()
