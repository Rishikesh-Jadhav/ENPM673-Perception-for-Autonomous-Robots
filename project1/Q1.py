# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video
cap = cv2.VideoCapture('ball.mov')

# Initialize empty lists to store x and y coordinates of the ball
x_coordinates = []
y_coordinates = []

# Define the lower and upper boundaries of the red color in HSV
lower_red = np.array([0, 110, 110])
upper_red = np.array([4, 255, 255])

# Loop through the frames in the video
while cap.isOpened():
    # Read the frame and convert it to HSV color space
    ret, frame = cap.read()
    if ret:
        # Get height and width of the frame
        height, width, _ = frame.shape
        
        # Define region of interest (ROI) to track the ball
        roi = frame[30:height, 50:width-55]

        # Apply Gaussian blur to remove noise
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)

        # Convert the ROI to HSV color space
        hsv = cv2.cvtColor(cv2.rotate(blurred, cv2.ROTATE_180), cv2.COLOR_BGR2HSV)
        
        # Threshold the image to extract the red color
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Get the average intensity of the mask
        avg_intensity = np.mean(mask)
    
        # Find the x,y coordinates of the maximum value in the grayscale image
        max_location = np.where(mask == np.max(mask))
        x, y = max_location[1][0], max_location[0][0]

        # Append the x and y coordinates to the respective lists
        if x & y != 0:
            x_coordinates.append(x)
            y_coordinates.append(y)
        
        # Display the mask in a window
        mask = cv2.flip(mask, 1) 
        mask = cv2.flip(mask, 0)
        cv2.imshow('ROI', mask)

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Plot the x and y coordinates of the ball
plt.plot(x_coordinates, y_coordinates, 'bo')
plt.xlim(1250.0, 0)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph of X and Y')
plt.show()

# Convert the x and y coordinates to arrays and remove zeros
x = np.array(x_coordinates)
y = np.array(y_coordinates)
x = x[np.nonzero(x)]
y = y[np.nonzero(y)]

# Create the matrix A
A = np.vstack([x**2, x, np.ones(len(x))]).T

# Create the observation matrix b
b = y

# Compute the least squares solution
x_least_sq = np.linalg.inv(A.T @ A) @ A.T @ b

# Print the equation of the curve
print("Equation of the Curve: y = {:.9f}x^2 + {:.9f}x {:.9f}".format(x_least_sq[0], x_least_sq[1],x_least_sq[2]))

# Plot the data and the fitted curve
plt.scatter(x, y)
plt.xlim(1250.0, 1.0)
plt.plot(x, A @ x_least_sq, color='red')
plt.show()

# Adjust the x-intercept of the fitted curve based on the starting point of the ball
x_least_sq[2]=x_least_sq[2]-y[0]-300+421
discriminant = x_least_sq[1]**2 - 4*x_least_sq[0]*x_least_sq[2]

# check if roots are real or imaginary
if discriminant < 0:
    print("Imaginary Roots")
else:
    # calculate the roots
    root1 = (-x_least_sq[1] + np.sqrt(discriminant)) / (2*x_least_sq[0]) #discarded as the value is negative
    root2 = (-x_least_sq[1] - np.sqrt(discriminant)) / (2*x_least_sq[0])
    print("X-coordinate of landing point of the ball is:" )
    print(root2)

























# def quadratic(x, a, b, c):
#     return a * x**2 + b * x + c

# def error(params, x, y):
#     return np.sum(np.power(quadratic(x, *params) - y, 2))


# # Initial guess for parameters
# params_guess = [1, 1, 1]

# # Minimize error function
# result = optimize.minimize(error, params_guess, args=(x, y))

# # Extract best-fit parameters
# a_fit, b_fit, c_fit = result.x


# # Construct the equation of the curve
# equation = f'y = {a_fit}x^2 + {b_fit}x + {c_fit}'
# print(equation)
# # Plot data points
# plt.scatter(x, y)

# # Generate x-values for the best-fit curve
# x_fit = np.linspace(x.min(), x.max(), 100)

# # Calculate y-values for the best-fit curve
# y_fit = quadratic(x_fit, a_fit, b_fit, c_fit)

# # Plot best-fit curve
# plt.plot(x_fit, y_fit, 'r-')
# plt.xlim(1200,0)
# # Display plot
# plt.show()




# coor = []

# coor = np.append(coor, [[x_co_ordinates,y_co_ordinates]], axis=0)                # Appending center coordinates in array  


# def StandardLeastSquares(coor):
    
#     x = x_co_ordinates                                                # x-coordinates array
#     y = y_co_ordinates                                                # y-coordinates array           
    
#     # System of equations formed for parabolic fit, i.e., ax^2 + bx + c = y
#     x_sq = np.power(x,2)
#     A = np.stack((x_sq, x, np.ones((len(x)), dtype=int )), axis=1)
#     A_t = A.transpose()
#     A_tA = A_t.dot(A)
#     A_tY = A_t.dot(y)
#     x_bar = (np.linalg.inv(A_tA)).dot(A_tY)                     
#     res = A.dot(x_bar)                                           # Output after applying Least squares model 
    
#     return res

# y = StandardLeastSquares(coor)

# # Plot for Video 1
# plt.subplot(121)
# plt.title('Video 1')
# plt.xlabel('time')
# plt.ylabel('position')
# plt.plot(coor[:,0], coor[:,1],'bo', label = 'Detected ball center')
# plt.plot(coor[:,0],y, c='red', label = 'Least Squares')
# plt.legend()





# # Read in the x and y values of the plotted points
# x = np.asarray(x_co_ordinates)
# y = np.asarray(y_co_ordinates)

# # Fit a quadratic polynomial to the data points
# coeffs = np.polyfit(x, y, 2)
# a, b, c = coeffs

# # Construct the equation of the curve
# equation = f'y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}'
# print(equation)

# # Evaluate the fitted curve at a range of x values
# x_range = np.linspace(x[0], x[-1], 100)
# y_range = a * x_range ** 2 + b * x_range + c

# # Plot the fitted curve on top of the original graph
# plt.scatter(x, y, label='Data Points')
# plt.plot(x_range, y_range,'r-' ,label='Fitted Curve')
# plt.xlim(1200,0)
# plt.legend()
# plt.show()



