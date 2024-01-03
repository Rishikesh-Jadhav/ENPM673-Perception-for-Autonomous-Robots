# ENPM673-Perception-for-Autonomous-Robots
This repository serves as a record of my academic experience in ENPM673 during the Spring of 2023. It includes my solutions and code submissions for all projects. Each project is organized within its respective folder, complete with accompanying documentation and any necessary resources.

## 📚 Course Overview
The Perception course offers an in-depth exploration of Classic Computer Vision principles and fundamental deep learning techniques. The curriculum centers on the augmentation of autonomous systems, encompassing robots, autonomous cars, and smart cameras. 
These hands-on projcts experiences helped us understand practical applications, ranging from lane detection for autonomous driving to the intricate task of constructing 3D models from 2D images. The course provides a holistic understanding of perception in autonomous systems, fostering both theoretical knowledge and practical skills.

1. **Curve Fitting and Trend Analysis:**
   - Identify the optimal trend line for a set of data points through curve fitting.

2. **Image Feature Recognition:**
   - Recognize key features in images, including corners, edges, and straight lines.

3. **3D Object Estimation:**
   - Estimate 3D information of objects based on their 2D images.

4. **Object Motion Metrics:**
   - Calculate motion metrics for objects, encompassing speed and direction, using camera feeds.

5. **Camera Pose Estimation:**
   - Conduct camera pose estimation for spatial understanding.

6. **Basic Image-based Machine Learning:**
   - Apply fundamental machine learning techniques to tasks involving images.

The course structure is enriched with four distinct projects, each described below.

## 📄 Project List
- Click [here](https://github.com/Rishikesh-Jadhav/VelocityEstimation-OpticalFlow) access ENPM-673 Final Project .

### [Project 1](https://github.com/Rishikesh-Jadhav/ENPM673-Perception-for-Autonomous-Robots/blob/main/project1/Report.pdf): Ball Tracking and Covariance Matrix, LS, TLS, and RANSAC implementaions for 3D Point Cloud

#### **Implementation and Learnings from Project 1**:

#### 1. Ball Tracking

Implemented ball tracking to follow the trajectory of a red ball thrown against a wall.

##### Steps:

1. Video captured using `cv2.VideoCapture`, and frames processed in a loop.
2. Color channels converted from BGR to HSV using `cv2.cvtColor`.
3. Red color channel isolated using `cv2.inRange` with specified upper and lower thresholds.
4. Pixel coordinates of the ball's center calculated by finding the mean of x and y coordinates.
5. Best-fit curve determined using the least squares method for pixel coordinates.

##### 1.1 Least Squares Method

Utilized the least squares method to find the best-fit curve (parabola) by minimizing mean square error.

#### 2. Covariance Matrix, LS, TLS, and RANSAC for 3D Point Cloud

Explored methods for fitting surfaces to 3D point cloud data.

##### 2.1 Covariance Matrix and Surface Normal

Calculated covariance matrix and determined surface normal's direction and magnitude using eigenvalues and eigenvectors.

##### 2.2 Standard Least Squares Method for 3D Point Cloud

Applied standard least squares method to find the best-fit surface plane.

##### 2.3 Total Least Squares Method for 3D Point Cloud

Used total least squares method to find the best-fit plane by minimizing error orthogonal to the plane.

##### 2.4 RANSAC Method

Implemented RANSAC for robust surface fitting, handling outliers in the data.

Number of iterations calculated using a formula involving probability values.

##### Observations and Interpretation of Results

- Total least squares method outperformed least squares method, especially in noisy data.
- RANSAC demonstrated superior accuracy in generating models, especially with outlier rejection.

#### 3. Observations and Interpretation of Results

- Total least squares method excels with noisy data.
- RANSAC consistently produces accurate models, especially with proper tuning of parameters.
- RANSAC is the preferred choice for outlier rejection and accurate model generation.

#### 4. Problems Encountered

1. Challenges in determining threshold limits for ball tracking.
2. Handling cases where the red channel was not filtered during ball tracking.
3. Issues with eigen vector assignment in total least squares method.
4. Error during RANSAC due to probability values resulting in a denominator of zero.
5. Complexity of RANSAC algorithm required referencing multiple examples and increasing iterations to reduce fluctuations.   

  
### [Project 2](https://github.com/Rishikesh-Jadhav/ENPM673-Perception-for-Autonomous-Robots/blob/main/project2/rjadhav1_proj2.pdf): Camera Pose Estimation and Image Stitching

#### **Implementaion and Learnings from Project 2**:

#### 1. Camera Pose Estimation using Homography

In this task, camera pose estimation was performed using homography on a video, involving the following steps:

##### 1.1 Image Processing Pipeline

1. Read video frame by frame.
2. Grayscale the image.
3. Blur the image.
4. Apply Thresholding to extract white color.
5. Perform Canny edge detection.
6. Use Hough transform algorithm on the frame.
7. Find peaks in the Hough space.
8. Draw lines corresponding to the Hough peaks.
9. Find the intersections between the detected lines.
10. Compute the homography matrix between the camera and the ground plane.
11. Decompose the homography matrix to obtain rotation and translation.

##### 1.2 Explanation and Results

The homography equation is used to describe the transformation between two images taken from different viewpoints. Steps involve Hough transformation for corner detection, homography computation, and homography decomposition.

#### 2. Image Stitching for Panoramic View

This task focused on stitching four images together to create a panoramic view:

##### 2.1 Pipeline

1. Load the four input images.
2. Convert images to grayscale.
3. Extract features using ORB or SIFT.
4. Match features using Brute-Force Matcher.
5. Visualize matched features.
6. Compute homographies between pairs of images.
7. Combine images using computed homographies.
8. Warp second image onto the first using OpenCV.
9. Repeat for the next pair until all four images are stitched.
10. Save the final panoramic image.

##### 2.2 Explanation and Results

The process involves feature extraction, feature matching, homography computation, and image blending. Homography is computed using RANSAC, and stitching involves warping and blending images.

#### 3. Problems Encountered and Solutions

Several challenges were faced during the project:

1. Determining Canny edge detection values.
2. Difficulty in detecting edges without using built-in functions.
3. Tricky aspects in finding camera rotation and translation.
4. Challenges in stitching due to dimension mismatches and homography application.   
    
### [Project 3](https://github.com/Rishikesh-Jadhav/ENPM673-Perception-for-Autonomous-Robots/blob/main/project3/rjadhav1_proj3.pdf): Camera calibration  

#### **Learnings from Project 3**:  

#### 1. Camera Calibration: Mathematical Approach

##### 1.1 Pipeline

Camera calibration corrects distortions and imperfections, providing accurate images. The pipeline involves:

1. Capture checkerboard images for calibration.
2. Determine world coordinates of checkerboard corners and find corresponding image coordinates.
3. Calculate camera parameters using the P matrix.
4. Extract the Rotation Matrix and Translation vector from the P matrix.
5. Find Reprojection error for each point.

##### 1.2 Results

1. Minimum number of matching points needed is 6 for mathematical calibration.
2. Mathematical formulation involves decomposing the P matrix and finding intrinsic matrix K, rotation matrix R, and translation vector T.
3. Intrinsic Matrix K:

   [-6.7912331e + 01, -7.9392768e − 02, 3.3562042e + 01;

   0, 6.7619034e + 01, 2.5845427e + 01;

   0, 0, 4.1946620e − 02]

5. Projection matrix P:
   
   [28.7364445 -1.75735415 -70.0687538 756.890519;

   -20.1369011 65.889012 -22.2140404 213.263797;

   -0.0277042391 -0.00259559759 -0.0313888009 1.00000000]
   
7. Rotation matrix R.

   [-0.74948643 0.11452983 -0.65203758;

   0.0453559 0.99149078 0.12202001;

   0.66046418 0.06187859 -0.74830349]
   
9. Translation vector T:

   [0.64862355;

    0.30183152;

    0.69751919;

    0.04064735]
   
11. Reprojection errors: [0.2856, 0.9726, 1.0361, 0.4541, 0.1909, 0.3190, 0.1959, 0.3083]

#### 2. Camera Calibration: Practical Approach

The objective is to calibrate the camera using real-world images.

##### 2.1 Pipeline

1. Read calibration images.
2. Grayscale and resize images.
3. Find corners using `cv2.findChessboardCorners()`.
4. Draw corners on images.
5. Calibrate using `cv2.calibrateCamera()` to obtain intrinsic parameters.
6. Compute reprojection error for each image.
7. Extract the camera matrix.

##### 2.2 Results

Corners detected in images, and reprojection errors:

- Reprojection errors: [0.1198, 0.2610, 0.4094, 0.5418, 0.2219, 0.3537, 0.0520, 0.2247, 0.4810, 0.4042, 0.4810, 0.5137, 0.4297]
- Intrinsic Matrix K:

[2.2317e + 03, 0, 7.7812e + 02;

0, 2.4542e + 03, 1.3235e + 03;

0, 0, 1.0000]

#### 3. Problems Encountered 

1. Determining correct K matrix in the mathematical approach.
2. Handling very low values in the K matrix.


### [Project 4](https://github.com/Rishikesh-Jadhav/ENPM673-Perception-for-Autonomous-Robots/blob/main/project4/rjadhav1_proj4.pdf): 

- **Learnings from Project 4**:

  
## Additional Resources
- [Course related resources](https://academiccatalog.umd.edu/graduate/courses/enpm/)


