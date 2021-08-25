import argparse
from read_data import read_coordinates
from model_dcgan import Discriminator, Generator
import torch
import random
import os, datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
import time

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", type=int, default=16, help="hidden filter channels")
parser.add_argument("--drop_out", type=float, default=0.5, help="drop out percentage")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning_rate")
parser.add_argument("--epochs", type=int, default=20000, help="epochs")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
args = parser.parse_args()

# hyperparameters
lr = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
hidden_size = args.hidden_size
drop_out = args.drop_out
loss_function = torch.nn.BCELoss()

# fixed random generator seed
torch.manual_seed(10)

current_dir = os.getcwd()

# create results file
if not os.path.isdir(os.path.join(current_dir, 'results')):
    os.mkdir(os.path.join(current_dir, 'results'))

# create models file
if not os.path.isdir(os.path.join(current_dir, 'models')):
    os.mkdir(os.path.join(current_dir, 'models'))

# create temporary result file
result_file = os.path.join(current_dir + '/results/', 'lr' + str(lr) + 'hd' + str(hidden_size) + 'bt' + str(batch_size) + 'dp' + str(drop_out) + 'ep' + str(epochs))
os.mkdir(result_file)

train_data = read_coordinates()
train_data = torch.nn.functional.normalize(train_data)
random.shuffle(train_data)
# max_value = torch.max(torch.abs(train_data))
print(train_data.shape)

# print('Normalize by(max tensor value):', max_value)
# train_data /= max_value

# record for TensorBoard
writer = SummaryWriter()

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

# create discriminator and generator
discriminator = Discriminator(hidden_size, train_data_pos, drop_out).to(device=device)
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
                plt.savefig(result_file + '/' + str(epoch) + '.png')

                torch.save(generator.model, './models/epoch_' + str(epoch) + '_model.h5')
writer.flush()
writer.close()

end = time.time()
print('Training duration:', (end - start) / 1000)
