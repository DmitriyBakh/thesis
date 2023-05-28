import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from datetime import datetime

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
num_classes = 2
input_dim = 1024
eps = 0.01
fake_probs = np.arange(0, 0.6, 0.1)
num_samples = 300

# transform = transforms.Compose(
#     [transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))])

transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])

# Load CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)

# Filter for only cats (class 3) and dogs (class 5)
trainset.targets = torch.tensor(trainset.targets)

# Get indices of cat and dog samples
cat_indices = (trainset.targets == 3).nonzero(as_tuple=True)[0]
dog_indices = (trainset.targets == 5).nonzero(as_tuple=True)[0]

# Check if there are enough samples, if not, use all samples
num_cat_samples = min(len(cat_indices), num_samples // 2)
num_dog_samples = min(len(dog_indices), num_samples // 2)

# Randomly select num_samples/2 from each class
cat_indices = cat_indices[torch.randperm(len(cat_indices))[:num_cat_samples]]
dog_indices = dog_indices[torch.randperm(len(dog_indices))[:num_dog_samples]]

# Combine the indices and use these to create your balanced training set
indices = torch.cat((cat_indices, dog_indices))

# Shuffle the combined indices
indices = indices[torch.randperm(len(indices))]

trainset.data = trainset.data[indices]
trainset.targets = trainset.targets[indices]

# Set labels for training data (0 for cats, 1 for dogs)
trainset.targets = (trainset.targets == 5).long()

testset.targets = torch.tensor(testset.targets)
cat_dog_test_indices = (testset.targets == 3) | (testset.targets == 5)
testset.data = testset.data[cat_dog_test_indices]
testset.targets = testset.targets[cat_dog_test_indices] # 0 for cats, 1 for dogs in testing data

# Set labels for testing data (0 for cats, 1 for dogs)
testset.targets = (testset.targets == 5).long()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=2)


# Define a simple neural network with one hidden layer
class Net(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)  # input layer to hidden layer
        # self.fc2 = nn.Linear(hidden_size, hidden_size)  # hidden layer #1
        self.fc2 = nn.Linear(hidden_size, 1)  # hidden layer #2 to output layer

    def forward(self, x):
        x = x.view(-1, input_dim)
        x = torch.relu(self.fc1(x)) # ReLU applied after the first fully connected layer
        # x = torch.relu(self.fc2(x)) # ReLU applied after the second fully connected layer
        # x = self.fc2(x)
        x = torch.sigmoid(self.fc2(x))  # Apply sigmoid to the output
        
        return x


# Initialize weights with Gaussian distribution
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # nn.init.normal_(m.weight)
        nn.init.zeros_(m.bias)

# Function to introduce fake labels
def introduce_fake_labels(labels, prob=0.00):
    num_fake = int(prob * len(labels))
    fake_indices = np.random.choice(len(labels), num_fake, replace=False)
    for idx in fake_indices:
        labels[idx] = 1 - labels[idx]  # flip the label
    return labels


results = {'train': {}, 'test': {}}

for fake_prob in fake_probs:
    start_time_prob = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    # for num_parameters in np.logspace(start=3, stop=5, num=10):
    # for num_parameters in np.linspace(1000, 70000, 12):
    for num_parameters in np.linspace(1000, 45000, 11):
        start_time_num_param = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        # hidden_size = int((num_parameters - num_classes) / (1 + input_dim + num_classes))
        hidden_size = int(num_parameters)
        net = Net(input_dim, hidden_size)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net.to(device)
        
        net.apply(weights_init)

        chk = net.state_dict()
        torch.save(chk, folder_path + f'chk_{int(fake_prob * 10)}_{hidden_size}_untrained.pt')

        # Use mean squared error loss and gradient descent
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.95)

        # Train the network
        for epoch in range(num_epochs):  # loop over the dataset multiple times
        # for epoch in range(2):  # loop for the test            
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels.float()
                labels = labels.view(-1, 1)

                # introduce fake labels
                introduce_fake_labels(labels, prob=fake_prob)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if int(fake_prob * 10) not in results['train']:
                results['train'][int(fake_prob * 10)] = {}

            results['train'][int(fake_prob * 10)][hidden_size] = {'epoch': epoch, 'loss': running_loss / len(trainloader)}

            if running_loss / len(trainloader) < eps:
                break

        print(f'Finished training for parameters: {hidden_size}, start: {start_time_num_param}, end: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}')

        with open(folder_path + 'results.pkl', 'wb') as fp:
            pickle.dump(results, fp)
            print('dictionary saved successfully to file')

        # Save the model
        chk = net.state_dict()
        torch.save(chk, folder_path + f'chk_{int(fake_prob * 10)}_{hidden_size}.pt')

        # Test the network on the test data
        correct = 0
        total = 0
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
                if int(fake_prob * 10) not in results['test']:
                    results['test'][int(fake_prob * 10)] = {}

                results['test'][int(fake_prob * 10)][hidden_size] = {'loss': loss.item()}
        
    print(f'Finished training for fake probability: {fake_prob}, start: {start_time_prob}, end: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}')

    with open(folder_path + 'results.pkl', 'wb') as fp:
        pickle.dump(results, fp)
        print('dictionary saved successfully to file')


with open(folder_path + 'results.pkl', 'wb') as fp:
    pickle.dump(results, fp)
    print('dictionary saved successfully to file')


with open(folder_path + 'results.pkl', 'rb') as fp:
    results = pickle.load(fp)
    print(results)
