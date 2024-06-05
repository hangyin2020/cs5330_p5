import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# class definitions
class MyNetwork(nn.Module):
    def __init__(self, num_filters, filter_sizes, hidden_nodes):
        super(MyNetwork, self).__init__()
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.hidden_nodes = hidden_nodes

        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # grayscale input
        for num_filter, filter_size in zip(num_filters, filter_sizes):
            conv_layer = nn.Conv2d(in_channels, num_filter, kernel_size=filter_size)
            self.conv_layers.append(conv_layer)
            in_channels = num_filter

        # Define pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate flattened size
        self.flattened_size = self._calculate_flattened_size()

        # Define fully connected layers
        fc_layers = []
        prev_layer_size = self.flattened_size
        for hidden_node in hidden_nodes:
            fc_layers.append(nn.Linear(prev_layer_size, hidden_node))
            prev_layer_size = hidden_node
        self.fc_layers = nn.ModuleList(fc_layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_nodes[-1], 10)

        # Dropout layer
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
            x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, self.flattened_size)
        print(f"flatten size is: {self.flattened_size}")
        for fc_layer in self.fc_layers:
            x = F.relu(fc_layer(x))
        return F.log_softmax(x, dim=1)

    def _calculate_flattened_size(self):
        # Assuming input image size is (28, 28)
        input_size = (28, 28)
        for filter_size in self.filter_sizes:
            input_size = ((input_size[0] - filter_size + 1) // 2, (input_size[1] - filter_size + 1) // 2)
        return input_size[0] * input_size[1] * self.num_filters[-1]

# useful functions with a comment for each function
def train_network( epoch,network,train_dataloader,optimizer,log_interval,train_losses, train_counter):
    network.train()

    for batch_idx, (data, target) in enumerate(train_dataloader):

        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                       100. * batch_idx / len(train_dataloader), loss.item()))

            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_dataloader.dataset)))
    return

def test_network(test_dataloader,network,test_losses, accuracies):
    print(f"Len of train_dataloader = {len(test_dataloader)}")
    print(f"Len of batch_size = {test_dataloader.batch_size}")
    print(f"Len of dataset = {len(test_dataloader.dataset)}")
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)
    print("testlose_len:{}".format(len(test_losses)))
    accuracies.append((100. * correct / len(test_dataloader.dataset)))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_dataloader.dataset),
    100. * correct / len(test_dataloader.dataset)))

def plot_losses(accuracy_counter, accuracies):
    plt.plot(accuracy_counter, accuracies, color='green')
    plt.scatter(accuracy_counter, accuracies, color='green')
    plt.legend(['Accuracy'], loc='upper left')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Accuracy')
    plt.show()


def run_network(train_dataloader,n_epochs,test_dataloader,network,optimizer, log_interval):
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_dataloader.dataset) for i in range(n_epochs + 1)]
    accuracies = []
    accuracy_counter = [i * len(train_dataloader.dataset) for i in range(n_epochs + 1)]

    test_network(test_dataloader, network, test_losses, accuracies)
    for epoch in range(1, n_epochs + 1):
        train_network(epoch, network, train_dataloader, optimizer, log_interval, train_losses, train_counter)
        test_network(test_dataloader, network, test_losses, accuracies)

    plot_losses(train_counter, train_losses)


# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv
    # Load MNIST dataset
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # define hyperparameters
    n_epochs = 2
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_dataloader = DataLoader(training_data, batch_size=batch_size_train)
    test_dataloader = DataLoader(test_data, batch_size=batch_size_test)

    # Plot the first six example digits from the test set
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 2
    for i in range(1, cols * rows + 1):
        img, label = test_data[i - 1]  # Test data starts from index 0
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

    # variation networks
    filters_lists = [[10]]
    filter_size_lists = [[5, 5]]
    hidden_nodes_lists = [[50, 10]]

    # Loop over configurations
    for filters_list, filter_size_list, hidden_nodes_list in zip(filters_lists, filter_size_lists, hidden_nodes_lists):
        # Instantiate network
        network = MyNetwork(filters_list, filter_size_list, hidden_nodes_list)
        print(f"network is: {network}")
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

        train_dataloader = DataLoader(training_data, batch_size=batch_size_train)
        test_dataloader = DataLoader(test_data, batch_size=batch_size_test)

        # Run training and testing for the networks
        run_network(train_dataloader, n_epochs, test_dataloader, network, optimizer, log_interval)


    # main function code
    return

if __name__ == "__main__":
    main(sys.argv)