import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import Generator

generator = Generator(10, 44)
generator.model = torch.load('model.h5')

latent_space_samples = torch.randn(1, 10).cuda()
generated_samples = generator(latent_space_samples).cpu()
generated_samples = generated_samples.view(generated_samples.shape[1], generated_samples.shape[2])
print(generated_samples)

x = generated_samples[:, 0].detach().numpy()
y = generated_samples[:, 1].detach().numpy()
z = generated_samples[:, 2].detach().numpy()

ax = plt.axes(projection='3d')  # 用這個繪圖物件建立一個Axes物件(有3D座標)
ax.set_xlabel('x label')
ax.set_ylabel('z label')
ax.set_zlabel('y label')  # 給三個座標軸註明
ax.scatter3D(x, z, y, color='Orange')
ax.scatter3D(0, 0, 0, color='Blue')
plt.show()