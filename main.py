import torch
import torch.nn as nn
import torch.optim as optim

# Define the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)  # Hidden layer 1
        self.fc3 = nn.Linear(64, 32)  # Hidden layer 2
        self.fc4 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input images
        x = torch.relu(self.fc1(x))  # ReLU activation function on layer 1
        x = torch.relu(self.fc2(x))  # ReLU activation function on layer 2
        x = torch.relu(self.fc3(x))  # ReLU activation function on layer 3
        x = torch.sigmoid(self.fc4(x))  # Sigmoid activation function on the output layer
        return x

# Instantiate the network
net = Net()

# Define a Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # Stochastic Gradient Descent

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# Dummy training loop (assuming trainloader is an iterator that provides batches of data and corresponding labels)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# Now, let's use the trained network for prediction
# For the purpose of this example, let's assume we have a DataLoader `testloader` for our test data
correct = 0
total = 0

# Since we're not training, we don't need to calculate the gradients
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)  # Pass the image through the network
        predicted = (outputs > 0.5).float()  # Since it's binary classification, we consider outputs > 0.5 as class 1 predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))


outputs = net(images)  # Pass the images through the network
predicted = (outputs > 0.5).float()  # If the output is > 0.5, consider it class 1, otherwise class 0
