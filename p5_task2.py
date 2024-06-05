"""

    Wenqing Fan
    Hang Yin
    cs5330 24sp project 5

    task 2 Examine your network

    We plot the weights of the first layer and then show the effect of the filters

"""


import torch
import cv2
import matplotlib.pyplot as plt
from p5_task1abcd import Net
from p5_task1abcd import load_test


# Visualize the weights of the first convolutional layer
def show_weights(layer1_weights):
    
    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(layer1_weights[i, 0].detach().numpy(),
                cmap='viridis', interpolation='none')
        plt.title("Filter: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# Visualize the effect of the layer of 10 filters on the first example
def show_effect(layer1_weights, example_data):
    # Get the first training example image
    image = example_data[0][0].numpy()  # Convert the tensor to a NumPy array

    # Create a figure to display the filtered images
    fig = plt.figure(figsize=(12, 8))

    # Iterate through each filter
    for i in range(10):
        # Extract the ith filter
        filter_kernel = layer1_weights[i, 0].detach().numpy()

        # Apply the filter to the image using filter2D
        filtered_image = cv2.filter2D(image, -1, filter_kernel)

        # Plot the filtered image
        plt.subplot(5, 4, i*2 + 1)
        plt.imshow(layer1_weights[i, 0].detach().numpy(),
                cmap='gray', interpolation='none')
        plt.axis('off')

        plt.subplot(5, 4, i*2 + 2)

        plt.imshow(filtered_image, cmap='gray', interpolation='none')
        # plt.title("Filter {}".format(i+1))
        plt.axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()


# main function to show the weights of first layer and the effects
def main(argv):
    test_loader = load_test(1000)

    network = Net()
    network.load_state_dict(torch.load('/Users/hangyin/results/model.pth'))
    print(network)

    # print(network.conv1.weight)
    layer1_weights = network.conv1.weight
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    show_weights(layer1_weights)
    show_effect(layer1_weights, example_data)
    return


if __name__ == "__main__":
    import sys
    main(sys.argv)
    