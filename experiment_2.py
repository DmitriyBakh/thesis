import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from datetime import datetime

import torch.nn.functional as F

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Define a transform to convert images to grayscale and to tensors
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Load the CIFAR-10 train and test datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Choose 5 random classes and assign them label 1, the rest label 0
chosen_classes = np.random.choice(10, 5, replace=False)

def relabel_dataset(dataset, chosen_classes):
    for i in range(len(dataset)):
        if dataset[i][1] in chosen_classes:
            dataset[i] = (dataset[i][0], 1)
        else:
            dataset[i] = (dataset[i][0], 0)

relabel_dataset(trainset)
relabel_dataset(testset)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 512)  # CIFAR-10 images are 32x32 pixels
        self.fc2 = nn.Linear(512, 2)  # binary classification

    def forward(self, x):
        x = x.view(-1, 32 * 32)  # flatten image input
        x = F.relu(self.fc1(x))  # hidden layer with ReLU activation
        x = self.fc2(x)  # output layer
        return x

# Instantiate the network
net = Net()



def train(net, trainloader, criterion, optimizer):
    epochs = 0
    while True:  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            labels = labels.float()  # convert labels to float for MSE loss
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(trainloader)
        print(f'Epoch {epochs + 1}, loss: {average_loss}')
        
        epochs += 1
        if average_loss < 0.01 or epochs >= 6000:
            break


def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))



criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

for _ in range(20):  # 20 experiments
    # Reassign labels for each experiment
    chosen_classes = np.random.choice(10, 5, replace=False)
    relabel_dataset(trainset, chosen_classes)
    relabel_dataset(testset, chosen_classes)
    
    train(net, trainloader, criterion, optimizer)
    test(net, testloader)
