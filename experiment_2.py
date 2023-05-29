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

# # Choose 5 random classes and assign them label 1, the rest label 0 TODO: DELETE
# chosen_classes = np.random.choice(10, 5, replace=False)

def relabel_dataset(dataset, chosen_classes, samples_per_class):
    new_dataset = []
    class_counts = {0: 0, 1: 0}
    
    for i in range(len(dataset)):
        label = 1 if dataset[i][1] in chosen_classes else 0
        if class_counts[label] < samples_per_class:
            new_dataset.append((dataset[i][0], label))
            class_counts[label] += 1
            
    return new_dataset


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


def balance_dataset(dataset, number_of_samples):
    class_counts = [0]*10  # Count for each of the 10 classes
    samples_per_class = number_of_samples // 10  # Equal distribution among classes
    new_dataset = []

    for image, label in dataset:
        if class_counts[label] < samples_per_class:
            new_dataset.append((image, label))
            class_counts[label] += 1

    return new_dataset


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, #TODO: DELETE
#                                           shuffle=True, num_workers=2)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

number_of_samples = 5000  # for example

for _ in range(20):  # 20 experiments
    # Balance the dataset
    balanced_trainset = balance_dataset(trainset, number_of_samples)

    # Reassign labels for each experiment
    chosen_classes = np.random.choice(10, 5, replace=False)
    relabel_dataset(balanced_trainset, chosen_classes)
    relabel_dataset(testset, chosen_classes)

    trainloader = torch.utils.data.DataLoader(balanced_trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    train(net, trainloader, criterion, optimizer)
    test(net, testloader)
