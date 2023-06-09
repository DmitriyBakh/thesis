# MNIST dataset
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pickle
from datetime import datetime

from model import Net, weights_init

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

def filter_digits(dataset):
    # Filter the dataset to include only 0 and 1 digits
    indices = (dataset.targets == 0) | (dataset.targets == 1)
    dataset.data = dataset.data[indices]
    dataset.targets = dataset.targets[indices]
    return dataset, indices


def balance_digits(dataset, num_samples):
    targets_0 = (dataset.targets == 0)
    targets_1 = (dataset.targets == 1)
    count_0 = targets_0.sum().item()
    count_1 = targets_1.sum().item()
    count_min = min(count_0, count_1, num_samples // 2)

    indices_0 = torch.where(targets_0)[0][:count_min]
    indices_1 = torch.where(targets_1)[0][:count_min]
    indices = torch.cat((indices_0, indices_1))

    dataset.data = dataset.data[indices]
    dataset.targets = dataset.targets[indices]

    return dataset, indices


# Function to introduce fake labels
def introduce_fake_labels(labels, prob=0.00):
    num_fake = int(prob * len(labels))
    fake_indices = np.random.choice(len(labels), num_fake, replace=False)
    for idx in fake_indices:
        labels[idx] = 1 - labels[idx]  # flip the label
    return labels


def make_training_dataset(num_samples, batch_size, fake_prob):
    # Download and load the training data
    mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    mnist_trainset, _ = filter_digits(mnist_trainset)
    mnist_trainset, _ = balance_digits(mnist_trainset, num_samples)

    # Introduce fake labels into the training set
    mnist_trainset.targets = introduce_fake_labels(mnist_trainset.targets, prob=fake_prob)

    # Create new indices for the SubsetRandomSampler
    indices = torch.randperm(len(mnist_trainset))

    # Create data loader with random sampling
    train_loader = DataLoader(mnist_trainset, batch_size=batch_size, sampler=SubsetRandomSampler(indices))

    return train_loader


# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)
# root_dir = "/content/gdrive/MyDrive/"

# folder_path = f'{root_dir}/data/'
folder_path = ''

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters and training setup.
num_epochs = 6000
batch_size = 100
learning_rate = 1e-3
learning_rate2 = 0.0001
input_dim = 784
eps = 0.01
fake_probs = np.arange(0, 0.6, 0.1)
num_samples = 6000
depth = 5
min_params, max_params, num_params = 1000, 37000, 14

# Download and load the test data
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

mnist_testset, _ = filter_digits(mnist_testset)

# Create data loaders with random sampling
testloader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)

results = {'train': {}, 'test': {}}

for fake_prob in fake_probs:
    start_time_prob = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    trainloader = make_training_dataset(num_samples, batch_size, fake_prob)
    for num_parameters in np.linspace(min_params, max_params, num_params):
        start_time_num_param = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        # hidden_size = int((num_parameters - num_classes) / (1 + input_dim + num_classes))
        hidden_size = int(num_parameters)
        net = Net(input_dim, hidden_size, depth)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net.to(device)
        
        net.apply(weights_init)

        chk = net.state_dict()
        torch.save(chk, folder_path + f'chk_{int(fake_prob * 10)}_{hidden_size}_untrained.pt')

        # Use mean squared error loss and gradient descent
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), 
                            lr=learning_rate if num_parameters < 25000 else learning_rate2,
                            momentum=0.95)
        # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=learning_rate, total_iters=num_epochs-1000)

        # Train the network
        for epoch in range(num_epochs):  # loop over the dataset multiple times
        # for epoch in range(2):  # loop for the test            
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels.float()
                labels = labels.view(-1, 1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # scheduler.step()

            if int(fake_prob * 10) not in results['train']:
                results['train'][int(fake_prob * 10)] = {}

            if hidden_size not in results['train'][int(fake_prob * 10)]:
                results['train'][int(fake_prob * 10)][hidden_size] = {}

            # Compute gradient norm
            grad_norm = 0.0
            for param in net.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item()
            
            mu_boundary = grad_norm / (running_loss / len(trainloader))

            results['train'][int(fake_prob * 10)][hidden_size][epoch] = {'loss': running_loss / len(trainloader), 'grad_norm': grad_norm, 'mu_boundary': mu_boundary}

            if running_loss / len(trainloader) < eps:
                break

        print(f'Finished training for parameters: {hidden_size}, start: {start_time_num_param}, end: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}')

        # Save the model
        chk = net.state_dict()
        torch.save(chk, folder_path + f'chk_{int(fake_prob * 10)}_{hidden_size}.pt')

        # Test the network on the test data
        correct = 0
        total = 0
        lossess = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                labels = labels.float()
                labels = labels.view(-1, 1)
                outputs = net(images)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, predicted)
                lossess.append(loss.item())
    
        if int(fake_prob * 10) not in results['test']:
            results['test'][int(fake_prob * 10)] = {}

        results['test'][int(fake_prob * 10)][hidden_size] = {'loss': sum(lossess) / len(lossess),
                                                              'accuracy': correct / total}
        with open(folder_path + 'results.pkl', 'wb') as fp:
            pickle.dump(results, fp)
            print('dictionary saved successfully to file')
    
    print(f'Finished training for fake probability: {fake_prob}, start: {start_time_prob}, end: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}')

    with open(folder_path + 'results.pkl', 'wb') as fp:
        pickle.dump(results, fp)
        print('dictionary saved successfully to file')


with open(folder_path + 'results.pkl', 'wb') as fp:
    pickle.dump(results, fp)
    print('dictionary saved successfully to file')


with open(folder_path + 'results.pkl', 'rb') as fp:
    results = pickle.load(fp)
    # print(results)
