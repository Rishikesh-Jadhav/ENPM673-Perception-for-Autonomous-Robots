# ENPM673-Perception-for-Autonomous-Robots
This repository serves as a record of my academic journey in ENPM673 during the Spring of 2023. It includes my solutions and code submissions for all projects. Each project has its dedicated folder with accompanying documentation and resources.

## 📚 Course Overview
The Perception course delves into Classic Computer Vision principles and fundamental deep learning techniques. The curriculum emphasizes enhancing autonomous systems like robots, self-driving cars, and smart cameras. Hands-on projects cover practical applications, such as lane detection and constructing 3D models from 2D images. The course aims to provide a comprehensive understanding of perception in autonomous systems, blending theoretical knowledge with practical skills.

1. **Curve Fitting and Trend Analysis:**
   - Optimal trend line identification for a set of data points through curve fitting.

2. **Image Feature Recognition:**
   - Recognition of key features in images, including corners, edges, and straight lines.

3. **3D Object Estimation:**
   - Estimation of 3D information for objects based on their 2D images.

4. **Object Motion Metrics:**
   - Calculation of motion metrics for objects, covering speed and direction using camera feeds.

5. **Camera Pose Estimation:**
   - Conducting camera pose estimation for spatial understanding.

6. **Basic Image-based Machine Learning:**
   - Application of fundamental machine learning techniques to image-related tasks.

The course structure includes four distinct projects, each outlined below.

## 📄 Project List
- Click [here](https://github.com/Rishikesh-Jadhav/VelocityEstimation-OpticalFlow) access ENPM-673 Final Project .

### [Project 1](https://github.com/Rishikesh-Jadhav/ENPM673-Perception-for-Autonomous-Robots/blob/main/project1/Report.pdf): Ball Tracking and Covariance Matrix, LS, TLS, and RANSAC implementaions for 3D Point Cloud

#### **Implementation and Learnings from Project 1**:

1. **Ball Tracking** : Implemented ball tracking to follow the trajectory of a red ball thrown against a wall
   - Video captured using `cv2.VideoCapture`, and frames processed in a loop.
   - Color channels converted from BGR to HSV using `cv2.cvtColor`.
   - Red color channel isolated using `cv2.inRange` with specified upper and lower thresholds.
   - Pixel coordinates of the ball's center calculated by finding the mean of x and y coordinates.
   - Best-fit curve determined using the least squares method for pixel coordinates.

   - **Least Squares Method:** Utilized the least squares method to find the best-fit curve (parabola) by minimizing mean square error.

Utilized the least squares method to find the best-fit curve (parabola) by minimizing mean square error.

2. **Covariance Matrix, LS, TLS, and RANSAC for 3D Point Cloud** : Explored methods for fitting surfaces to 3D point cloud data.

   - **Covariance Matrix and Surface Normal:** Calculated covariance matrix and determined surface normal's direction and magnitude using eigenvalues and eigenvectors.
   - **Standard Least Squares Method for 3D Point Cloud:** Applied standard least squares method to find the best-fit surface plane.
   - **Total Least Squares Method for 3D Point Cloud:** Used total least squares method to find the best-fit plane by minimizing error orthogonal to the plane.
   - **RANSAC Method:** Implemented RANSAC for robust surface fitting, handling outliers in the data.

   - **Observations and Interpretation of Results:**
     - Total least squares method outperformed least squares method, especially in noisy data.
     - RANSAC demonstrated superior accuracy in generating models, especially with outlier rejection.

   - **Problems Encountered:**
     1. Challenges in determining threshold limits for ball tracking.
     2. Issues with eigen vector assignment in the total least squares method.
     3. Error during RANSAC due to probability values resulting in a denominator of zero.
     4. Complexity of RANSAC algorithm required referencing multiple examples and increasing iterations to reduce fluctuations.

  
### [Project 2](https://github.com/Rishikesh-Jadhav/ENPM673-Perception-for-Autonomous-Robots/blob/main/project2/rjadhav1_proj2.pdf): Camera Pose Estimation and Image Stitching

#### **Implementaion and Learnings from Project 2**:

1. **Camera Pose Estimation using Homography**
   - In this task, camera pose estimation was performed using homography on a video, involving the following steps:

   - **Image Processing Pipeline:**
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

   - **Explanation and Results:**
     - The homography equation is used to describe the transformation between two images taken from different viewpoints. Steps involve Hough transformation for corner detection, homography computation, and homography decomposition.

2. **Image Stitching for Panoramic View**
   - This task focused on stitching four images together to create a panoramic view:

   - **Pipeline:**
     1. Load the four input images.
     2. Convert images to grayscale.
     3. Extract features using ORB or SIFT.
     4. Match features using Brute-Force Matcher.
     5. Visualize matched features.
     6. Compute homographies between pairs of images.
     7. Combine images using computed homographies.
     8. Warp the second image onto the first using OpenCV.
     9. Repeat for the next pair until all four images are stitched.
     10. Save the final panoramic image.

   - **Explanation and Results:**
     - The process involves feature extraction, feature matching, homography computation, and image blending. Homography is computed using RANSAC, and stitching involves warping and blending images.

   - **Problems Encountered and Solutions:**
     1. Determining Canny edge detection values.
     2. Difficulty in detecting edges without using built-in functions.
     3. Tricky aspects in finding camera rotation and translation.
     4. Challenges in stitching due to dimension mismatches and homography application.

  
### [Project 3](https://github.com/Rishikesh-Jadhav/ENPM673-Perception-for-Autonomous-Robots/blob/main/project3/rjadhav1_proj3.pdf): Camera Calibration  

#### **Implementation and Learnings from Project 3**:  

1. **Camera Calibration: Mathematical Approach**

   - **Pipeline:**
     1. Capture checkerboard images for calibration.
     2. Determine world coordinates of checkerboard corners and find corresponding image coordinates.
     3. Calculate camera parameters using the P matrix.
     4. Extract the Rotation Matrix and Translation vector from the P matrix.
     5. Find Reprojection error for each point.

   - **Results:**

      1. Minimum number of matching points needed is 6 for mathematical calibration.
      2. Mathematical formulation involves decomposing the P matrix and finding intrinsic matrix K, rotation matrix R, and translation vector T.
      3. Intrinsic Matrix K:
      
         ```plaintext
         [-6.7912331e + 01, -7.9392768e − 02, 3.3562042e + 01;
          0, 6.7619034e + 01, 2.5845427e + 01;
          0, 0, 4.1946620e − 02]
         ```
      
      5. Projection matrix P:
         
         ```plaintext
         [28.7364445 -1.75735415 -70.0687538 756.890519;
          -20.1369011 65.889012 -22.2140404 213.263797;
          -0.0277042391 -0.00259559759 -0.0313888009 1.00000000]
         ```
         
      7. Rotation matrix R.
      
         ```plaintext
         [-0.74948643 0.11452983 -0.65203758;
          0.0453559 0.99149078 0.12202001;
          0.66046418 0.06187859 -0.74830349]
         ```
         
      9. Translation vector T:
      
         ```plaintext
         [0.64862355;
          0.30183152;
          0.69751919;
          0.04064735]
         ```
         
      11. Reprojection errors:
          ```[0.2856, 0.9726, 1.0361, 0.4541, 0.1909, 0.3190, 0.1959, 0.3083] ```


2. **Camera Calibration: Practical Approach**

   - The objective is to calibrate the camera using real-world images.

   - **Pipeline:**
     1. Read calibration images.
     2. Grayscale and resize images.
     3. Find corners using `cv2.findChessboardCorners()`.
     4. Draw corners on images.
     5. Calibrate using `cv2.calibrateCamera()` to obtain intrinsic parameters.
     6. Compute reprojection error for each image.
     7. Extract the camera matrix.

   - **Results:**

      - Corners detected in images, and reprojection errors:

        - Reprojection errors: ```plaintext
          [0.1198, 0.2610, 0.4094, 0.5418, 0.2219, 0.3537, 0.0520, 0.2247, 0.4810, 0.4042, 0.4810, 0.5137, 0.4297]
          ```
        - Intrinsic Matrix K:

          ```plaintext
          [2.2317e + 03, 0, 7.7812e + 02;
           0, 2.4542e + 03, 1.3235e + 03;
           0, 0, 1.0000]
          ```

#### 3. Problems Encountered 

1. Determining correct K matrix in the mathematical approach.
2. Handling very low values in the K matrix.


### [Project 4](https://github.com/Rishikesh-Jadhav/ENPM673-Perception-for-Autonomous-Robots/blob/main/project4/rjadhav1_proj4.pdf): Stereo Vision and Depth Perception

#### **Implementation and Learnings from Project 4**:

   - The fourth project in my perception course involved addressing four sub-tasks, each contributing to the overall goal of stereo vision:

1. **Calibration Pipeline :**
   - Utilized ORB feature extraction to find matching features in stereo images.
   - Estimated the Fundamental matrix and Essential matrix, considering camera intrinsics.
   - Decomposed Essential matrix into translation and rotation.

2. **Rectification Pipeline :**
   - Applied perspective transformation to rectify stereo images for easier comparison.
   - Computed homography matrices to map original to rectified image coordinates.
   - Visualized rectification effects through epipolar lines and feature points overlay.

3. **Correspondence Pipeline :**
   - Implemented a correspondence pipeline involving matching windows and disparity calculation.
   - Generated grayscale and color heat maps for visualizing disparity.

4. **Image Depth Computation Pipeline :**
   - Calculated depth values from a disparity map, considering camera calibration parameters.
   - Produced grayscale and color heat maps for depth visualization.

#### 2. Results
The pipelines were applied to three datasets, yielding specific outcomes for each room:

- **Chess Room :**
  - Fundamental matrix and Essential matrix estimation.
  - Visual representation of matched features.

- **Ladder Room :**
  - Fundamental matrix and Essential matrix estimation.
  - Visual representation of matched features.

- **Art Room :**
  - Fundamental matrix and Essential matrix estimation.
  - Visual representation of matched features.

#### 3. Problems Encountered and Solutions

1. **Calibration Outliers :**
   - Difficulty in removing outliers during camera calibration.
   - Tricky estimation of the Fundamental matrix.

2. **Rectification Issues :**
   - Inability to achieve horizontal epipolar lines during rectification.
   - Warping difficulties.

3. **Correspondence Challenges :**
   - Issues arising from problems in the previous processes.
   - Formulaic challenges in implementing correspondence.
  
## Additional Resources
- [Course related resources](https://academiccatalog.umd.edu/graduate/courses/enpm/)


