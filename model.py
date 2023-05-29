import torch
import torch.nn as nn

# Initialize weights with Gaussian distribution
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # nn.init.normal_(m.weight)
        nn.init.zeros_(m.bias)


# Define a simple neural network with one hidden layer
class Net(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)  # input layer to hidden layer
        # self.fc2 = nn.Linear(hidden_size, hidden_size)  # hidden layer #1
        self.fc2 = nn.Linear(hidden_size, 1)  # hidden layer #2 to output layer

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = torch.relu(self.fc1(x)) # ReLU applied after the first fully connected layer
        # x = torch.relu(self.fc2(x)) # ReLU applied after the second fully connected layer
        x = self.fc2(x)
        # x = torch.sigmoid(self.fc2(x))  # Apply sigmoid to the output
        
        return x