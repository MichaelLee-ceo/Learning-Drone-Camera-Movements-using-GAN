from torch import nn

class Discriminator(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super().__init__()
        self.inputSize = inputSize
        self.model = nn.Sequential(
            nn.Conv1d(inputSize, hiddenSize, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(hiddenSize, hiddenSize * 2, 3, 1, 1, bias=False),
            nn.BatchNorm1d(hiddenSize * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(hiddenSize * 2, hiddenSize * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(hiddenSize * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #
            # nn.Conv1d(hiddenSize * 4, hiddenSize * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm1d(hiddenSize * 8),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(hiddenSize * 4, 1, 3, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(output.shape[0], -1)
        # print('Discriminator output shape:', output, output.shape)
        return output

class Generator(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.outputSize = outputSize
        self.model = nn.Sequential(
            nn.ConvTranspose1d(inputSize, hiddenSize * 8, 3, 1, 0, bias=False),
            nn.BatchNorm1d(hiddenSize * 8),
            nn.ReLU(True),

            nn.ConvTranspose1d(hiddenSize * 8, hiddenSize * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(hiddenSize * 4),
            nn.ReLU(True),
            #
            nn.ConvTranspose1d(hiddenSize * 4, hiddenSize * 2, 3, 1, 1, bias=False),
            nn.BatchNorm1d(hiddenSize * 2),
            nn.ReLU(True),
            #
            nn.ConvTranspose1d(hiddenSize * 2, outputSize, 3, 1, 1, bias=False),
            nn.ReLU(True),
        )

    def forward(self, x):
        output = self.model(x)
        # print('Generator outputs shape:', output.shape)
        output = output.view(x.size(0), self.outputSize, 3)
        return output