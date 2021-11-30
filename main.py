import torch
from read_data import read_coordinates
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model_dcgan import Generator
from trajectory_animation import track
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--frame_skip", type=int, default=5, help="skipped interval for frames in the trajectories")
args = parser.parse_args()

# hyperparameters
frame_skip = args.frame_skip

# fixed random generator seed
torch.manual_seed(10)


lr = 0.001
batch_size = 4
epochs = 20000
hidden_sizes =  [8]
drop_outs = [0.25, 0.5, 0.7]


latent_space_samples = torch.randn((1, 1, 100))

for hidden_size in hidden_sizes:
    for drop_out in drop_outs:
        train_setting = 'lr' + str(lr) + 'hd' + str(hidden_size) + 'bt' + str(batch_size) + 'dp' + str(drop_out) + 'ep' + str(epochs)

        for epoch in range(11000, epochs + 1, 1000):
            generator = Generator(100, hidden_size, 38)
            generator.model = torch.load('./models/' + train_setting + '/' + train_setting + '_' + str(epoch) + '_model.h5', map_location=('cpu'))

            generated_samples = generator(latent_space_samples).cpu()
            generated_samples = generated_samples.view(generated_samples.shape[1], generated_samples.shape[2])

            print('Generated_samples shape:', generated_samples.shape)
            input()
            # print(generated_samples, generated_samples.shape)

            # x = generated_samples[::frame_space, 0].detach().numpy()
            # y = generated_samples[::frame_space, 1].detach().numpy()
            # z = generated_samples[::frame_space, 2].detach().numpy()

            # ax = plt.axes(projection='3d')  # 用這個繪圖物件建立一個Axes物件(有3D座標)
            # ax.set_xlabel('x label')
            # ax.set_ylabel('z label')
            # ax.set_zlabel('y label')  # 給三個座標軸註明
            #
            # ax.scatter3D(x, z, y, color='Orange')
            # ax.scatter3D(0, 0, 0, color='Blue')
            # plt.show()

            track(generated_samples, frame_skip, train_setting, epoch)
