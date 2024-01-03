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

### [Project 1](https://github.com/Rishikesh-Jadhav/ENPM673-Perception-for-Autonomous-Robots/blob/main/project1/Project1.pdf): Rendering with Pytorch3D

- **Implementation and Learnings from Project 1**:

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

- **[Project 1 Report](https://github.com/Rishikesh-Jadhav/ENPM673-Perception-for-Autonomous-Robots/blob/main/project1/Report.pdf)**    
  
### [Project 2](https://github.com/Rishikesh-Jadhav/CMSC848F-3D-Vision/tree/main/Assignment2):  Single View to 3D

- **Learnings from Project 2**:
- **[Project 2 Report](#)**    
 
    
### [Project 3](https://github.com/Rishikesh-Jadhav/CMSC848F-3D-Vision/tree/main/Assignment3):  Volume Rendering and Neural Radiance Fields

- **Learnings from Project 3**:
- **[Project 3 Report](#)**    


### [Project 4](https://github.com/Rishikesh-Jadhav/CMSC848F-3D-Vision/tree/main/Assignment4): Point Cloud Classification and Segmentation

- **Learnings from Project 4**:
- **[Project 4 Report](#)**    

  
## Additional Resources
- [Course related resources](https://academiccatalog.umd.edu/graduate/courses/enpm/)


