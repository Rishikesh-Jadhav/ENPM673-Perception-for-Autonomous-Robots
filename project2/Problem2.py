import cv2
import numpy as np
import matplotlib.pyplot as plt


def perspective_transform(src_points, homography_matrix):
    # Reshaping the source points to remove the middle dimension
    src_points = np.squeeze(src_points)
    homogenous_src_points = np.hstack((src_points, np.ones((src_points.shape[0], 1))))

    # Computing the dot product between the homography matrix and the homogenous source points
    transformed_points_homogenous = np.dot(homography_matrix, homogenous_src_points.T).T

    # Dividing by the third element of each row to convert the result back to non-homogenous coordinates
    transformed_points = transformed_points_homogenous[:, :2] / transformed_points_homogenous[:, 2:]

    return transformed_points

def findhomography(src_pts,dst_pts):
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

img1 = cv2.imread('image_1.jpg')
img2 = cv2.imread('image_2.jpg')
img3 = cv2.imread('image_3.jpg')
img4 = cv2.imread('image_4.jpg')

# Converting images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

#Creating ORB object
orb = cv2.ORB_create()

# Extracting features from each image
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
kp3, des3 = orb.detectAndCompute(gray3, None)
kp4, des4 = orb.detectAndCompute(gray4, None)

# Matching features between each consecutive image
bf = cv2.BFMatcher()
matches12 = bf.match(des1, des2)
matches23 = bf.match(des2, des3)
matches34 = bf.match(des3, des4)

# Visualizing the matches between each consecutive image
img12 = cv2.drawMatches(img1, kp1, img2, kp2, matches12, None, flags=2)
img23 = cv2.drawMatches(img2, kp2, img3, kp3, matches23, None, flags=2)
img34 = cv2.drawMatches(img3, kp3, img4, kp4, matches34, None, flags=2)

plt.figure(figsize=(20, 10))
plt.subplot(131), plt.imshow(img12), plt.title('Matches between Image 1 and Image 2')
plt.subplot(132), plt.imshow(img23), plt.title('Matches between Image 2 and Image 3')
plt.subplot(133), plt.imshow(img34), plt.title('Matches between Image 3 and Image 4')
plt.show()

# Computing the homographies between each pair of images
src_pts12 = np.float32([kp1[m.queryIdx].pt for m in matches12]).reshape(-1, 1, 2)
dst_pts12 = np.float32([kp2[m.trainIdx].pt for m in matches12]).reshape(-1, 1, 2)
H12, _ = cv2.findHomography(src_pts12, dst_pts12, cv2.RANSAC, 5.0)

src_pts23 = np.float32([kp2[m.queryIdx].pt for m in matches23]).reshape(-1, 1, 2)
dst_pts23 = np.float32([kp3[m.trainIdx].pt for m in matches23]).reshape(-1, 1, 2)
H23, _ = cv2.findHomography(src_pts23, dst_pts23, cv2.RANSAC, 5.0)

src_pts34 = np.float32([kp3[m.queryIdx].pt for m in matches34]).reshape(-1, 1, 2)
dst_pts34 = np.float32([kp4[m.trainIdx].pt for m in matches34]).reshape(-1, 1, 2)
H34, _ = cv2.findHomography(src_pts34, dst_pts34,cv2.RANSAC, 5.0)

h, w = img1.shape[:2]

corners1 = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

corners2 = perspective_transform(corners1, H12)

x_min = min(corners2[:, 0].min(), 0)
y_min = min(corners2[:, 1].min(), 0)
x_max = max(corners2[:, 0].max(), w)
y_max = max(corners2[:, 1].max(), h)
out_w = int(x_max - x_min)
out_h = int(y_max - y_min)

trans_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

pano12 = cv2.warpPerspective(img1, trans_mat.dot(H12), (out_w, out_h))
y_min = int(y_min)
h = int(h)
x_min = int(x_min)
w = int(w)
pano12[-y_min:-y_min+h, -x_min:-x_min+w] = img2

corners3 = perspective_transform(corners2, H23)


x_min = min(corners3[:, 0].min(), 0)
y_min = min(corners3[:, 1].min(), 0)
x_max = max(corners3[:, 0].max(), out_w)
y_max = max(corners3[:, 1].max(), out_h)
out_w = int(x_max - x_min)
out_h = int(y_max - y_min)

trans_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
pano23 = cv2.warpPerspective(pano12, trans_mat.dot(H23), (out_w, out_h))
y_min = int(y_min)
h = int(h)
x_min = int(x_min)
w = int(w)
pano23[-y_min:h-y_min, -x_min:w-x_min] = img3


corners4 = perspective_transform(corners3, H34)

x_min = min(corners4[:, 0].min(), 0)
y_min = min(corners4[:, 1].min(), 0)
x_max = max(corners4[:, 0].max(), out_w)
y_max = max(corners4[:, 1].max(), out_h)
out_w = int(x_max - x_min)
out_h = int(y_max - y_min)


trans_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
pano34 = cv2.warpPerspective(pano23, trans_mat.dot(H34), (out_w, out_h))
y_min = int(y_min)
h = int(h)
x_min = int(x_min)
w = int(w)
pano34[-y_min:h-y_min, -x_min:w-x_min] = img4


plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(pano34, cv2.COLOR_BGR2RGB))
plt.title('Final Panorama')
plt.show()


