from torch import nn

class Discriminator(nn.Module):
    def __init__(self, hiddenSize, frame_num, drop_rate):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(3, hiddenSize * 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop_rate),

            nn.Conv1d(hiddenSize * 16, hiddenSize * 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop_rate),

            nn.Conv1d(hiddenSize * 8, hiddenSize * 4, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop_rate),

            nn.Conv1d(hiddenSize * 4, hiddenSize * 2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop_rate),

            nn.Flatten(),

            nn.Linear(hiddenSize * 2 * frame_num, hiddenSize * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop_rate),
            nn.Linear(hiddenSize * 2, hiddenSize),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop_rate),
            nn.Linear(hiddenSize, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.shape[0], 3, -1)
        output = self.model(x)
        output = output.view(output.shape[0], -1)
        # print('Discriminator output shape:', output.shape)
        return output

class Generator(nn.Module):
    def __init__(self, latent_dim, hiddenSize, outputSize):
        super().__init__()
        self.outputSize = outputSize
        self.model = nn.Sequential(
            nn.Conv1d(1, hiddenSize*2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize*2),
            nn.ReLU(),

            nn.Conv1d(hiddenSize*2, hiddenSize*4, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 4),
            nn.ReLU(),

            nn.Conv1d(hiddenSize*4, hiddenSize*8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 8),
            nn.ReLU(),

            nn.Conv1d(hiddenSize*8, hiddenSize*16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 16),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(hiddenSize*16*latent_dim, hiddenSize*8),
            nn.ReLU(),
            nn.Linear(hiddenSize*8, self.outputSize*3),
        )

    def forward(self, x):
        output = self.model(x)
        # print('Generator outputs shape:', output.shape)
        output = output.view(x.size(0), self.outputSize, 3)
        return output
