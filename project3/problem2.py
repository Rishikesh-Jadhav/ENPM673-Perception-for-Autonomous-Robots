# References :
#https://medium.com/analytics-vidhya/camera-calibration-with-opencv-f324679c6eb7

import numpy as np
import cv2
import os

#Size of each square in mm.(GIVEN)
square_size = 21.5

# Path to the folder containing calibration images.
path = 'Calibration_Imgs/'

# Number of corners in the checkerboard.
Nx = 9
Ny = 6

# Creating 3D coordinates of the chessboard corners
threeD_points = np.zeros((Nx*Ny, 3), np.float32)
threeD_points[:, :2] = np.mgrid[0:Nx, 0:Ny].T.reshape(-1, 2)
threeD_points *= square_size

corner_points = []
# Looping through each image, finding corners and storing them in the corner_points list.
for filename in os.listdir(path):
   
    # Loading the image.
    img = cv2.imread(os.path.join(path, filename))
    
    #converting the images to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # resizing images.
    img_resized = cv2.resize(gray, (0,0), fx=0.25, fy=0.25)

    # Finding the checkerboard corners using the findChessboardCorners method.
    ret, corners = cv2.findChessboardCorners(img_resized, (Nx, Ny), None)
    cv2.drawChessboardCorners(img_resized, (Nx, Ny), corners,ret)

    #displaying the images.
    cv2.imshow('New Image', img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # If corners are found, add them to the list of corner points.
    if ret == True:
        corner_points.append(corners)


# Calibrating the camera using the cv2.calibrateCamera method on the corner points found above.
img_shape = gray.shape[::-1]
ret, camera_matrix, distortion_coeff, rotvecs, transvecs = cv2.calibrateCamera([threeD_points]*len(corner_points), 
                                                   corner_points, 
                                                   img_shape, 
                                                   None, 
                                                   None)

# Computing reprojection error for each image using inbuilt cv2 functions
rpg_errors_list = []

# looping over the corner points list to find the reprojection error
for i in range(len(corner_points)):
    img_points, _ = cv2.projectPoints(threeD_points, rotvecs[i], transvecs[i], camera_matrix, distortion_coeff)
    error = cv2.norm(corner_points[i], img_points, cv2.NORM_L2) / len(img_points)
    rpg_errors_list.append(error)

    print("Reprojection error for image no.{} : {}".format(i, error))

# Extracting the camera matrix
k = camera_matrix[:3,:3]
print("The Final Camera matrix : \n {}".format(k))