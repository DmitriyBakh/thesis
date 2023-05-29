import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from datetime import datetime

import torch.nn.functional as F

from model import Net, weights_init


def relabel_dataset(dataset, chosen_classes, samples_per_class):
    new_dataset = []
    class_counts = {0: 0, 1: 0}
    
    for i in range(len(dataset)):
        label = 1 if dataset[i][1] in chosen_classes else 0
        if class_counts[label] < samples_per_class:
            new_dataset.append((dataset[i][0], label))
            class_counts[label] += 1
            
    return new_dataset


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

net = Net()

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

number_of_samples = 5000

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
