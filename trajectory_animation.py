import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from read_data import read_coordinates
import datetime
import os
import csv

dir = os.path.join(os.getcwd() + '/generated_positions/')
if not os.path.isdir(dir):
    print('Creating File:generated_positions')
    os.mkdir(dir)

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


def track(camera_positions, frame_space):
    # THE DATA POINTS
    # camera_positions = torch.load('./mediapipe_videos/coordinates_camera_pos/shot_7.pt').numpy()
    # train_data = read_coordinates()
    # camera_positions = train_data[4].numpy()
    x = camera_positions[::frame_space, 0].detach().numpy()
    y = camera_positions[::frame_space, 1].detach().numpy()
    z = camera_positions[::frame_space, 2].detach().numpy()

    # print(x, y, z)
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

    with open(dir + '/generated_position_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(camera_positions.tolist()[::frame_space])
        print('Saved generated trajectory into csv.')