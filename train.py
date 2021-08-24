from read_data import read_coordinates
from model_dcgan import Discriminator, Generator
import torch
import random
import os, datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter

dir = os.path.join(os.getcwd() + '/results/', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(dir)

# fixed random generator seed
torch.manual_seed(10)

train_data = read_coordinates()
train_data = torch.nn.functional.normalize(train_data)
random.shuffle(train_data)
# max_value = torch.max(torch.abs(train_data))

# print('Normalize by(max tensor value):', max_value)
# train_data /= max_value

# record for TensorBoard
writer = SummaryWriter()

print(train_data.shape)

# create training dataset
train_data_length, train_data_pos, train_data_dim = train_data.shape
print('Trained frames number', train_data_pos)
train_labels = torch.zeros(train_data_length)
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using:', device, 'for training')

# hyperparameters
lr = 0.001
epochs = 20000
batch_size = 4
hidden_size = 16
loss_function = torch.nn.BCELoss()

# create discriminator and generator
discriminator = Discriminator(hidden_size, train_data_pos).to(device=device)
generator = Generator(100, hidden_size, train_data_pos).to(device=device)

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)


for epoch in range(1, epochs+1):
    # print(epoch)
    for idx, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(device=device)

        latent_space_samples = torch.randn((batch_size, 1, 100)).to(device=device)
        generated_samples = generator(latent_space_samples)
        generated_labels = torch.zeros((batch_size, 1)).to(device=device)


        all_samples = torch.cat((real_samples, generated_samples))
        all_labels = torch.cat((real_samples_labels, generated_labels))

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 1, 100)).to(device=device)

        # Training the generator
        generator.zero_grad()
        output_generator = generator(latent_space_samples)
        output_discriminator_generated = discriminator(output_generator)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # Show training loss
        if epoch % 10 == 0 and idx == len(train_loader) - 1:
            print(f"\nEpoch: {epoch}, Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch}, Loss G.: {loss_generator}")

            writer.add_scalar("D: Loss/train", loss_discriminator, epoch)
            writer.add_scalar("G: Loss/train", loss_generator, epoch)

            if epoch % 1000 == 0:
                latent_samples = torch.randn((1, 1, 100)).cuda()
                generated_samples = generator(latent_samples).cpu()
                generated_samples = generated_samples.view(generated_samples.shape[1], generated_samples.shape[2])

                x = generated_samples[:, 0].detach().numpy()
                y = generated_samples[:, 1].detach().numpy()
                z = generated_samples[:, 2].detach().numpy()

                ax = plt.axes(projection='3d')  # 用這個繪圖物件建立一個Axes物件(有3D座標)
                ax.set_xlabel('x label')
                ax.set_ylabel('z label')
                ax.set_zlabel('y label')  # 給三個座標軸註明
                ax.scatter3D(x, z, y, color='Orange')
                ax.scatter3D(0, 0, 0, color='Blue')
                plt.savefig(dir + '/' + str(epoch) + '.png')

                torch.save(generator.model, './models/epoch_' + str(epoch) + '_model.h5')
writer.flush()
writer.close()
