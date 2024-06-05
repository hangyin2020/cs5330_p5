"""

    Wenqing Fan
    Hang Yin
    cs5330 24sp project 5

    task 4 Design your own experiment

    We replaced the MNIST data with FashionMNIST data

"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# load the training data


def load_train(batch_size_train):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(root="data", train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ])),
        batch_size=batch_size_train, shuffle=False)
    return train_loader

# load the test data


def load_test(batch_size_test):
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(root="data", train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ])),
        batch_size=batch_size_test, shuffle=False)
    return test_loader

# The plot to show the first 6 digits


def show_example(example_data, example_targets):
    # plot the first six example digits
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# the network class definition, including the construction of the neural network and the forward function


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 10 filters, 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 20 filters, 5x5

        # Dropout layer
        self.conv2_drop = nn.Dropout2d(p=0.5)  # Dropout with 50% probability

        # Fully connected layers
        self.fc1 = nn.Linear(320, 50)  # Flattened input from conv2, 50 nodes
        self.fc2 = nn.Linear(50, 10)  # 10 output classes

    def forward(self, x):
        # First convolution layer with ReLU and max pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 2x2 max pooling with ReLU
        # Second convolution layer with dropout, ReLU, and max pooling
        # 2x2 max pooling with ReLU and dropout
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten the output for fully connected layers
        x = x.view(-1, 320)  # Flatten to 320 dimensions
        # First fully connected layer with ReLU
        x = F.relu(self.fc1(x))  # 50-node fully connected layer with ReLU
        # Dropout layer
        # Dropout with 50% probability
        x = F.dropout(x, training=self.training)
        # Second fully connected layer
        x = self.fc2(x)  # Final fully connected layer
        return F.log_softmax(x, dim=1)  # Log softmax for output


# The train function for the model and store them into files
def train(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter):
    # set the model in training mode
    network.train()
    # the enumerate returns a iterator, we can use it to iterate through the data
    for batch_idx, (data, target) in enumerate(train_loader):
        # set the gradients to zero since PyTorch by default accumulates gradients
        optimizer.zero_grad()
        # output is a tensor that represents the predictions made by the network
        # example format:
        # output = torch.tensor([
        # [-1.2,  2.3, -0.5,  0.1, -0.9,  3.5,  1.2, -2.0,  0.6, -3.1],
        # [ 3.1,  0.2, -1.5, -0.8,  2.9, -0.3,  1.5, -2.8,  0.4, -1.9],
        # ...
        # # More rows representing predictions for each image in the batch
        # ])
        output = network(data)
        # computes the negative log likelihood loss between the predicted output and the target labels
        loss = F.nll_loss(output, target)
        # computes the gradients of the loss with respect to the model parameters
        loss.backward()
        # updates the model parameters based on the computed gradients.
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            # Save the model to a file
            torch.save(network.state_dict(),
                       './results/model.pth')
            torch.save(optimizer.state_dict(),
                       './results/optimizer.pth')

# The test function to check the accuracy of the model


def test(network, test_loader, test_losses):
    # set the model in testing mode
    network.eval()
    test_loss = 0
    correct = 0

    # turn off the gradient calculation here since we are only doing the test
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            # compute the negative log likelihood lose between output and target
            test_loss += F.nll_loss(output, target, size_average=False).item()
            print(f"test_loss is: {test_loss}")
            # the prediction of the index of the max values in dimension 1 of output tensor
            # dimension 0 is row and dimension 1 is column
            pred = output.data.max(1, keepdim=True)[1]
            # sum of all the correct prediction
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print(f"test_loss is: {test_loss}")
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# To plot the loss after training


def show_Loss(train_counter, train_losses, test_counter, test_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    # fig
    plt.show()

# To plot the prediction of the first 6 examples


def show_prediction(network, example_data):
    # labels map
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    with torch.no_grad():
        output = network(example_data)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            labels_map[output.data.max(1, keepdim=True)[1][i].item()]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# the main function for task 1 including training, testing and plots


def main(argv):
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader = load_train(batch_size_train)
    test_loader = load_test(batch_size_test)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    example_data.shape

    # show the first few example images from test
    show_example(example_data, example_targets)

    network = Net()
    optimizer = optim.SGD(network.parameters(),
                          lr=learning_rate, momentum=momentum)
    print(network)

    test(network, test_loader, test_losses)

    for epoch in range(1, n_epochs + 1):
        train(epoch, network, train_loader, optimizer,
              log_interval, train_losses, train_counter)
        test(network, test_loader, test_losses)

    # draw the plot to show the loss
    show_Loss(train_counter, train_losses, test_counter, test_losses)

    # draw the prediction of the first few examples
    show_prediction(network, example_data)

    return


if __name__ == "__main__":
    import sys
    main(sys.argv)
