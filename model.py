from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim * 3, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.LeakyReLU(),
            # nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.LeakyReLU(),
            # nn.Dropout(0.25),

            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim * 3)
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 2048),
            nn.ReLU(),
            # nn.Dropout(0.25),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, self.output_dim * 3),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), self.output_dim, 3)
        return output