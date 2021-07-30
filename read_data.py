import torch
import os

data_path = './mediapipe_videos/coordinates_camera_pos/'
fullpath = os.path.join(os.getcwd() + data_path)

data = []
for root, dirs, files in os.walk(fullpath):
    for idx, f in enumerate(files):
        if f.endswith('.pt'):
            data.append(torch.load(fullpath + f))

print(len(data))
minimun_coordinates = 0
for t in data:
    minimun_coordinates = min(len(t), minimun_coordinates)
    print(len(t), t.shape)

print(minimun_coordinates)
