from torch import nn

class Discriminator(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super().__init__()
        self.inputSize = inputSize
        self.model = nn.Sequential(
            nn.Conv1d(inputSize, hiddenSize * 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 8),
            # nn.MaxPool1d(kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

            nn.Conv1d(hiddenSize * 8, hiddenSize * 4, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 4),
            # nn.MaxPool1d(kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

            nn.Conv1d(hiddenSize * 4, hiddenSize * 2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 2),
            # nn.MaxPool1d(kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

            nn.Flatten(),

            nn.Linear(hiddenSize * 2 * 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        # output = output.view(output.shape[0], -1)
        # print('Discriminator output shape:', output, output.shape)
        return output

class Generator(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.outputSize = outputSize
        self.model = nn.Sequential(
            nn.Conv1d(inputSize, hiddenSize*8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize*2),
            # nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),

            nn.Conv1d(hiddenSize*8, hiddenSize*16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 4),
            nn.ReLU(),

            nn.Conv1d(hiddenSize*16, hiddenSize*32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 8),
            nn.ReLU(),

            nn.Conv1d(hiddenSize*32, hiddenSize*64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 16),
            nn.ReLU(),

            nn.Conv1d(hiddenSize*64, hiddenSize*128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(hiddenSize * 32),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(hiddenSize*128, self.outputSize*3),
        )

    def forward(self, x):
        output = self.model(x)
        # print('Generator outputs shape:', output.shape)
        output = output.view(x.size(0), self.outputSize, 3)
        return output