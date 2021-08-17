import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from read_data import read_coordinates

# References
# https://gist.github.com/neale/e32b1f16a43bfdc0608f45a504df5a84
# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
# https://riptutorial.com/matplotlib/example/23558/basic-animation-with-funcanimation

# ANIMATION FUNCTION
def func(num, dataSet, line):
    # NOTE: there is no .set_data() for 3 dim data...
    line.set_data(dataSet[0:2, :num])
    line.set_3d_properties(dataSet[2, :num])
    return line


# THE DATA POINTS
# camera_positions = torch.load('./mediapipe_videos/coordinates_camera_pos/shot_7.pt').numpy()
train_data = read_coordinates()
camera_positions = train_data[4].numpy()
x = camera_positions[:, 0]
y = camera_positions[:, 1]
z = camera_positions[:, 2]

print(x, y, z)
dataSet = np.array([x, z, y])
numDataPoints = camera_positions.shape[0]

# GET SOME MATPLOTLIB OBJECTS
fig = plt.figure()
ax = plt.axes(projection='3d')  # 用這個繪圖物件建立一個Axes物件(有3D座標)

# NOTE: Can't pass empty arrays into 3d version of plot()
ax.scatter3D(0, 0, 0, c='b')  # For line plot
line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='g')[0]  # For line plot


# AXES PROPERTIES]
# ax.set_xlim3d([limit0, limit1])
ax.set_xlabel('X')
ax.set_ylabel('z')
ax.set_zlabel('y')
ax.set_title('Trajectory of camera')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet, line), interval=100, blit=False)
# line_ani.save(r'AnimationNew.mp4')


plt.show()