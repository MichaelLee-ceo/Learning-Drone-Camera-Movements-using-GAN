from read_data import read_coordinates
from trajectory_animation import track

frame_skip = 1
train_setting = ''
file_path = 'training_data_animation'

train_data = read_coordinates('/coordinates_camera_pos_regression/')[:100]

for idx in range(len(train_data)):
    track(train_data[idx], frame_skip, train_setting, idx, file_path)