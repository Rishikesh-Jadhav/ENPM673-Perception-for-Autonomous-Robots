
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def standard_least_squares(x,y,z):
    
    x_matrix=np.vstack((x,y,z)).transpose()
    y_matrix=np.ones(x.shape)
    x_transpose_x=np.linalg.pinv(np.dot(x_matrix.transpose(),x_matrix))
    x_y=np.dot(x_matrix.transpose(),y_matrix)
    B=np.dot(x_transpose_x,x_y)
    return B

def eval_hypo(x_points,y_points,z_points,coef,threshold=0.1):
    distance=np.empty(1,)
    for i in range(x_points.shape[0]):
        #distance of hypothesis plane and points in Dataset
        dist=(np.abs((coef[0]*x_points[i])+(coef[1]*y_points[i]+(coef[2]*z_points[i])-1)))/np.sqrt(((coef[0]**2)+(coef[1]**2)+coef[2]**2))
        distance=np.append(distance,dist)

    success=np.where(distance<=threshold)[0].shape[0]
    return success

def RANSAC(x_points,y_points,z_points,out_prob=0.5,success_prob=0.99,sample_points=3):
    e=out_prob
    p=success_prob
    hypothesis=[]
    list_of_points=np.empty(0)
    inliers=np.empty(0)
    thresh_result=[]
    samples=int(np.log(1 - p) / np.log(1 - np.power((1 - e), sample_points)))
    samples=1100  
    for i in range(0,samples):
        points=np.random.randint(0,x_points.shape[0],(3,))
        list_of_points=np.append(list_of_points,points)
        x=x_points[points]
        y=y_points[points]
        z=z_points[points]
        coef=standard_least_squares(x,y,z)
        success=eval_hypo(x_points,y_points,z_points,coef)
        thresh_result.append(success)
        inliers=np.append(inliers,success)
        hypothesis.append(coef)

    best_hypothesis=np.argmax(inliers)
    return hypothesis[best_hypothesis],inliers[best_hypothesis] 



data = np.loadtxt('pc1.csv', delimiter=',')
data2 = np.loadtxt('pc2.csv', delimiter=',')

#reading data from frame 
x1 = data[:, 0]
y1 = data[:, 1]
z1 = data[:, 2]
x2 = data2[:, 0]
y2 = data2[:, 1]
z2 = data2[:, 2]

ransac_coef,inliers=RANSAC(x1,y1,z1)
ransac_coef_2,inliers_2=RANSAC(x2,y2,z2)

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

# Plot data and plane for pc1
xx1, yy1 = np.meshgrid(x1,y1)
zz1 = np.mean(z1)-(((ransac_coef[0]*(xx1-np.mean(x1)))+(ransac_coef[1]*(yy1-np.mean(y1)))) / (ransac_coef[2]))
ax1.plot_surface(xx1, yy1, zz1, alpha=0.1,color='blue')
ax1.scatter(x1,y1,z1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title("Ransac for pc2")

# Plot data and plane for pc2
xx2, yy2 = np.meshgrid(x2,y2)
zz2 = np.mean(z2)-(((ransac_coef_2[0]*(xx2-np.mean(x2)))+(ransac_coef_2[1]*(yy2-np.mean(y2)))) / (ransac_coef_2[2]))
ax2.plot_surface(xx2, yy2, zz2, alpha=0.1,color='blue')
ax2.scatter(x2,y2,z2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title("Ransac for pc2")

plt.show()

