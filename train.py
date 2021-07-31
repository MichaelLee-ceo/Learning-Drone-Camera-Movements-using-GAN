from read_data import read_coordinates
from model import Discriminator, Generator
import torch

train_data = read_coordinates()

# fixed random generator seed
torch.manual_seed(10)

# create training dataset
train_data_length, train_data_pos, train_data_dim = train_data.shape
train_labels = torch.zeros(train_data_length)
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using:', device, 'for training')

# create discriminator and generator
discriminator = Discriminator(train_data_pos).to(device=device)
generator = Generator(50, train_data_pos).to(device=device)

# hyperparameters
lr = 0.001
epochs = 300
batch_size = 8
loss_function = torch.nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

def train_gan():
    for epoch in range(epochs):
        for idx, (real_samples, _) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples = real_samples.to(device=device)
            real_samples_labels = torch.ones((batch_size, 1)).to(device=device)

            latent_space_samples = torch.randn((batch_size, 50)).to(device=device)
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
            latent_space_samples = torch.randn((batch_size, 50)).to(device=device)

            # Training the generator
            generator.zero_grad()
            output_generator = generator(latent_space_samples)
            output_discriminator_generated = discriminator(output_generator)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

            # Show training loss
            if epoch % 10 == 0 and idx == batch_size - 1:
                print(f"\nEpoch: {epoch}, Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch}, Loss G.: {loss_generator}")

    return generator