import torch
from read_data import read_coordinates
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model_dcgan import Generator
from trajectory_animation import track


generator = Generator(100, 16, 31)
generator.model = torch.load('./models/epoch_18000_model.h5')

latent_space_samples = torch.randn((1, 1, 100)).cuda()
# print(latent_space_samples, latent_space_samples.shape)
generated_samples = generator(latent_space_samples).cpu()
generated_samples = generated_samples.view(generated_samples.shape[1], generated_samples.shape[2])
# print(generated_samples, generated_samples.shape)

frame_space = 3

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

track(generated_samples, frame_space)