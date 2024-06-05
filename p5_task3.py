"""

    Wenqing Fan
    Hang Yin
    cs5330 24sp project 5

    task 3 Transfer Learning on Greek Letters
    included 3 of each of the greek letters that was hand written
    data can be found with the link in the report

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from p5_task1abcd import show_prediction
from p5_task1abcd import test
from p5_task1abcd import Net
from p5_task1abcd import show_Loss

# greek data set transform


class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)
    
class GreekHandTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/2000, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

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
                (batch_idx*5) + ((epoch-1)*len(train_loader.dataset)))


# main function to load the network and replace the last layer and
# then train and test the data
def main(argv):
    n_epochs = 20
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 1
    network = Net()
    network.load_state_dict(torch.load('/Users/hangyin/results/model.pth'))

    
    # freezes the parameters for the whole network
    for param in network.parameters():
        param.requires_grad = False

    # Add the new layer after loading the pre-trained network
    network.fc2 = nn.Linear(50, 3)  # New layer with 3 output nodes

    for param in network.fc2.parameters():
        param.requires_grad = True
    # network.fc2.requires_grad_ = True

    optimizer = optim.SGD(network.parameters(),
                          lr=learning_rate, momentum=momentum)

    print(network)

    training_set_path = '/Users/hangyin/Desktop/cs5330_24spring/project5/data/greek_train'
    hand_training_set_path = '/Users/hangyin/Desktop/cs5330_24spring/project5/data/greek_train_hand'


    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                             (0.1307,), (0.3081,))])),
        batch_size=5,
        shuffle=False)
    
    # DataLoader for the Greek data hand written set
    greek_train_hand = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(hand_training_set_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekHandTransform(),
                                                                                   torchvision.transforms.Normalize(
                                             (0.1307,), (0.3081,))])),
        batch_size=9,
        shuffle=False)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(greek_train.dataset) for i in range(n_epochs + 1)]



    test(network, greek_train, test_losses)

    for epoch in range(1, n_epochs + 1):
        train(epoch, network, greek_train, optimizer,
              log_interval, train_losses, train_counter)
        test(network, greek_train, test_losses)

    show_Loss(train_counter, train_losses, test_counter, test_losses)


    test_losses_hand = []


    test(network, greek_train_hand, test_losses_hand)

    examples = enumerate(greek_train_hand)
    batch_idx, (example_data, example_targets) = next(examples)


    show_prediction(network, example_data, 9, 3, 3)
    
    return


if __name__ == "__main__":
    import sys
    main(sys.argv)
