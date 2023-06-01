#MNIST dataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from datetime import datetime

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


def balance_dataset(dataset, number_of_samples):
    class_counts = [0]*10  # Count for each of the 10 classes
    samples_per_class = number_of_samples // 10  # Equal distribution among classes
    new_dataset = []

    for image, label in dataset:
        if class_counts[label] < samples_per_class:
            new_dataset.append((image, label))
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
        # print(f'Epoch {epochs + 1}, loss: {average_loss}')
        
        epochs += 1
        if average_loss < 0.01 or epochs >= 6000:
            break

    return epochs


def test(net, testloader, criterion):
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            labels = labels.float()  # convert labels to float for MSE loss
            loss = criterion(outputs, labels.view(-1, 1))
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

    average_loss = test_loss / len(testloader)
    accuracy = 100 * correct / total

    # print('Average test loss: %.2f' % average_loss)
    # print('Accuracy of the network on the test images: %d %%' % accuracy)

    return average_loss, accuracy


# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Define a transform to convert images to grayscale and to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the MNIST train and test datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)


hidden_size = 37000
depth = 5
num_epochs = 6000
batch_size = 100
learning_rate = 1e-3
learning_rate2 = 0.0001
input_dim = 784
eps = 0.01
fake_probs = np.arange(0, 0.6, 0.1)
num_samples = 6000
# min_params, max_params, num_params = 1000, 37000, 14

experiments_results = {
    'epoch_counts': [],
    'test_losses': [],
    'accuracies': []
}

criterion = nn.MSELoss()

for i in range(20):  # 20 experiments
    # Balance the dataset
    balanced_trainset = balance_dataset(trainset, num_samples)

    # Reassign labels for each experiment
    chosen_classes = np.random.choice(10, 5, replace=False)
    balanced_trainset = relabel_dataset(balanced_trainset, chosen_classes)
    testset = relabel_dataset(testset, chosen_classes)

    trainloader = torch.utils.data.DataLoader(balanced_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # for hidden_size in range(min_params, max_params+1):
    #     net = Net(input_dim, hidden_size, depth)
    #     optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.95)

    #     train(net, trainloader, criterion, optimizer)
    #     test(net, testloader)

    net = Net(input_dim, hidden_size, depth)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.95)

    epochs = train(net, trainloader, criterion, optimizer)
    experiments_results['epoch_counts'].append(epochs)

    test_loss, accuracy = test(net, testloader)
    experiments_results['test_losses'].append(test_loss)
    experiments_results['accuracies'].append(accuracy)

with open('experiment_2_b.pkl', 'wb') as f:
    pickle.dump(experiments_results, f)
    print('dictionary saved successfully to file')
