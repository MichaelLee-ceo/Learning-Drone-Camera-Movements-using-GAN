import torch
import os
import math
import numpy as np

def read_coordinates(data_path = '/mediapipe_videos/coordinates_camera_pos/'):
    fullpath = os.path.join(os.getcwd() + data_path)

    data = []
    data_length = []
    for root, dirs, files in os.walk(fullpath):
        for idx, f in enumerate(files):
            if f.endswith('.pt'):
                print('read coordinates file:', f)
                data.append(torch.load(fullpath + f))
                data_length.append(data[-1].shape[0])

    # print('Found data files:', len(data))

    # calculate mean coordinates from each video
    mean_data_count = math.floor(np.mean(data_length))
    # print('Mean data points:', mean_data_count)

    # amending data to match mean coordinates
    for i in range(len(data)):
        for j in range(data[i].shape[0], mean_data_count):
            data[i] = torch.cat((data[i], torch.unsqueeze(data[i][-1], 0)), 0)
        data[i] = data[i][:mean_data_count]

    # reverse trajectories
    # prev_data = data
    # data.reverse()
    # prev_data += data
    # print(lambda data:data[0])

    sorted_data = []
    for i in range(len(data)):
        sorted_data.append(torch.stack(sorted(data[i], key=lambda x: x[0])))

    return torch.stack(sorted_data)
