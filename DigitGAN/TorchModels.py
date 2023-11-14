import torch
from torch import nn, optim


class Generator(nn.Module):

    def __init__(self, *, input_size: int, hidden_dim: int, output_size: int, lr: float, beta1: float = 0.9):
        super(Generator, self).__init__()

        # Linear, batch, activation, dropout
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid(),
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1,  0.999))

    def forward(self, noise, label):
        x = torch.cat((noise, label), dim=-1)
        return self.model(x)

    def predict(self, noise, label):  # Returns (28*28) instead of (784)
        x = self.forward(noise, label).detach()
        return x.view(-1, 28, 28).squeeze()  # remove all shape 1 dimensions


class Discriminator(nn.Module):

    def __init__(self, *, input_size: int, hidden_dim: int, output_size: int, lr: float = 0.0002, beta1: float = 0.9):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid(),
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1,  0.999))

    def forward(self, image, label):
        x = torch.cat((image, label), dim=-1)
        return self.model(x).view(-1)


class GeneratorCNN(nn.Module):

    def __init__(self, *, input_size: int, hidden_dim: int, lr: float, beta1: float = 0.9):
        super(GeneratorCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim * 7 * 7),
            nn.BatchNorm1d(hidden_dim * 7 * 7),
            nn.ReLU(),

            # Reshape to (hidden_dim, 7, 7)
            nn.Unflatten(1, (hidden_dim, 7, 7)),

            # ConvTranspose2d layer 1, 14 x 14
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),

            # ConvTranspose2d layer 2, 1 x 28 x 28
            nn.ConvTranspose2d(hidden_dim // 2, 1,
                               kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1,  0.999))

    def forward(self, noise, label):
        x = torch.cat((noise, label), dim=-1)
        x = self.model(x).squeeze()
        return x

    def predict(self, noise, label):
        x = self.forward(noise, label).detach()
        return x


class DiscriminatorCNN(nn.Module):

    def __init__(self, *, num_classes: int, hidden_dim: int, lr: float = 0.0002, beta1: float = 0.9):
        super(DiscriminatorCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(1, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),

            # Convolutional Layer 2
            nn.Conv2d(hidden_dim, hidden_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2)
        )

        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 2 * 7 * 7 + num_classes, 1),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

        self.model = nn.Sequential(self.conv_layers, self.linear_layer)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1,  0.999))

    def forward(self, image, label):
        conv_output = self.conv_layers(image.view(-1, 1, 28, 28))
        conv_output = conv_output.view(conv_output.size(0), -1)
        combined_input = torch.cat((conv_output, label), dim=1)
        x = self.linear_layer(combined_input).view(-1)
        return x
