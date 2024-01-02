
# Import necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
# Loading video file
cap = cv2.VideoCapture('project2.avi')

# Initializing arrays for storing Hough space parameters
hough_x = []
hough_y = []

def canny(image):
    blur = cv2.GaussianBlur(image, (7,7), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 85, 255)
    return canny


# Defining a function for performing Hough transform
def hough_transform(frame, res=1, angle_step=1):
    w, h = frame.shape
    theta = np.deg2rad(np.arange(-90, 90, angle_step))
    rhp = np.ceil(int(math.sqrt(w**2 + h**2)))
    rhl = np.arange(-rhp, rhp+1, res)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    # Creating an accumulator matrix for storing Hough space values
    acc = np.zeros((len(rhl), len(theta)), dtype=np.uint64)
    yid, xid = np.nonzero(frame)
    for i in range(len(xid)):
        xn = xid[i]
        yn = yid[i]
        for j in range(len(theta)):
            rho = int(rhp + int(xn * costheta[j] + yn * sintheta[j]))
            acc[rho, j] += 1

    # Returning the accumulator matrix and associated rho and theta values
    return acc, rhl, theta

# Defining a function for identifying peaks in the Hough space
def find_hough_peaks(hough_acc, num_peaks, threshold=295, neighbourhood=2):
    
    indices = []
    # Iterating over the specified number of peaks to identify
    for i in range(num_peaks):
        # Finding the index of the maximum value in the accumulator matrix
        idx = np.argmax(hough_acc)
        hough_acc_idx = np.unravel_index(idx, hough_acc.shape)
        indices.append(hough_acc_idx)
        # Suppressing peaks in the neighborhood around the maximum value
        y, x = hough_acc_idx
        x_min = max(x - neighbourhood, 0)
        x_max = min(x + neighbourhood + 1, hough_acc.shape[1])
        y_min = max(y - neighbourhood, 0)
        y_max = min(y + neighbourhood + 1, hough_acc.shape[0])
        hough_acc[y_min:y_max, x_min:x_max] = 0
    # Returning the peak indices and the modified accumulator matrix
    return indices, hough_acc


def draw_lines(img, idx, rho, theta):
    color = (0, 0, 255)
    for i in range(len(idx)):
        r = rho[idx[i][0]]
        t = theta[idx[i][1]]
        if t==0 and t==np.pi:
            cv2.line(img, (r, 0), (r, img.shape[0]), (255, 0, 0), 2)
        else:
            m = -1 / np.tan(t)
            b = r / np.sin(t)
            y1 = 0
            x1 = int((y1 - b) / m)
            y2 = img.shape[0]
            x2 = int((y2 - b) / m)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


def hough_interesction_check(img,idx,rho,theta):
    cps=[]
    for i in range(len(idx)):
        rho1=rho[idx[i][0]]
        theta1=theta[idx[i][1]]
        for j in range(i+1,len(idx)):
            rho2=rho[idx[j][0]]
            theta2=theta[idx[j][1]]

            #find solution of form AX=B where A is set of angles and b is  rho in matrix form
            if theta1==theta2 and theta1 != 90+theta2:
                continue
            A=np.array([[np.cos(theta1),np.sin(theta1)],[np.cos(theta2),np.sin(theta2)]])
            B=np.array([rho1,rho2])
            try:
                x,y=np.linalg.solve(A,B)
            except np.linalg.LinAlgError:
                continue
            if (0 < x < img.shape[1]) and (0 < y < img.shape[0]):
                cps.append((int(x),int(y)))
    return cps


def homography(src_pts,dst_pts):
    A=[]
    for i in range(len(src_pts)):
        x, y = src_pts[i][0],src_pts[i][1]
        m, n = dst_pts[i][0], dst_pts[i][1]
        A.append([x, y, 1, 0, 0, 0, -m * x, -m * y, -m])
        A.append([0, 0, 0, x, y, 1, -n * x, -n * y, -n])
    A = np.array(A)
    u, s, v= np.linalg.svd(A)
    H_matrix = v[-1, :].reshape(3,3)
    H_matrix_normalized = H_matrix / H_matrix[2,2]
    return H_matrix_normalized


def decomposition(H):
    #Given Intrinsic matrix of the camera

    K = np.array([[0.00138, 0, 0.0946],
                  [0, 0.00138, 0.0527],
                  [0, 0, 1]])
    
    #Formula - H= K[R1 R2 R3 T] using this we can calculate R1 R2 R3 and t
    r1=np.dot(np.linalg.inv(K),H[:,0])
    r2=np.dot(np.linalg.inv(K),H[:,1])
    r3=np.cross(r1,r2)
    t=np.linalg.inv(K) @ H[:,2]
    rot=np.column_stack((r1,r2,r3))

    return rot,t

dst_points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

r = []
p = []
y = []
t_x = []
t_y = []
t_z  = []

while cap.isOpened():
    ret, frame = cap.read()
    if ret:


        # Getting edges from Processed image
        edges = canny(frame)

        # Applying the Hough transform to detect lines in the image
        accumulator, rhos, thetas = hough_transform(edges)
        peak_indexes, _ = find_hough_peaks(accumulator, 4, neighbourhood=1)
        draw_lines(frame, peak_indexes, rhos, thetas)
        intersection_pts = hough_interesction_check(edges, peak_indexes, rhos, thetas)
        

        # Computing the homography matrix to map the intersection points to the destination points
        homography_matrix = homography(intersection_pts, dst_points)
        rotation, translation = decomposition(homography_matrix)
        print("Rotation:", rotation,rotation.shape)
        print("Translation:", translation,translation.shape)

        pitch = -np.arcsin(rotation[2, 0])
        
        # Check for gimbal lock at pitch = +/-90 degrees
        if np.abs(pitch) == np.pi/2:
            # Gimbal lock has occurred, so we set roll to 0 and solve for yaw
            roll = 0
            yaw = np.arctan2(rotation[0, 1], rotation[1, 1])
        else:
            # Extract roll (around y-axis) and yaw (around z-axis)
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        r.append(np.rad2deg(roll))
        p.append(np.rad2deg(pitch))
        y.append(np.rad2deg(yaw))
        t_x.append(translation[0])
        t_y.append(translation[1])
        t_z.append(translation[2])
        cv2.imshow("Lines", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
    else:
        break

cap.release()
cv2.destroyAllWindows()

plt.plot(r,label="roll")
plt.plot(p,label="pitch")
plt.plot(y,label="Yaw")
plt.xlabel("Frame")
plt.ylabel("Angle(Degree)")
plt.legend()
plt.show()

plt.plot(t_x,label="T_x")
plt.plot(t_y,label="T_y")
plt.plot(t_z,label="T_z")
plt.xlabel("Frame")
plt.ylabel("Movement(CM)")
plt.legend()
plt.show()