"""

    Wenqing Fan
    Hang Yin
    cs5330 24sp project 5

    task 1 e Read the network and run it on the test set

    We followed the tutorial on https://nextjournal.com/gkoehler/pytorch-mnist
    for most of the process, and we organize the code into functions for further use

"""


import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from p5_task1abcd import Net
from p5_task1abcd import load_test
from p5_task1abcd import test

# plot the result for test samples
def show_test(example_data, output):
    fig = plt.figure()
    for i in range(9):
        # print the output for the current sample
        print(f"output {i}: {output[i]}")
        plt.subplot(3, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# main function to load the network and then run the test
def main(argv):
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    test_loader = load_test(batch_size_test)


    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)


    continued_network = Net()
    continued_optimizer = optim.SGD(continued_network.parameters(), lr=learning_rate,
                                    momentum=momentum)

    # load the trained network
    network_state_dict = torch.load('/Users/hangyin/results/model.pth')
    continued_network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load('/Users/hangyin/results/optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)
    
    test_losses = []

    # Run the test
    test(continued_network, test_loader, test_losses)

    with torch.no_grad():
        output = continued_network(example_data)

    show_test(example_data, output)
    return


if __name__ == "__main__":
    import sys
    main(sys.argv)
