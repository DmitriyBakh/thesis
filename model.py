import torch
import torch.nn as nn

# Initialize weights with Gaussian distribution
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # nn.init.normal_(m.weight)
        nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, input_dim, hidden_size, depth=2):
        super(Net, self).__init__()

        # Check that depth is at least 2
        if depth < 2:
            raise ValueError("Depth must be at least 2")

        self.input_dim = input_dim

        # Define the first layer
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())

        # Add hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Define the output layer
        layers.append(nn.Linear(hidden_size, 1))

        # Store the layers in a sequential module
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.module(x)
