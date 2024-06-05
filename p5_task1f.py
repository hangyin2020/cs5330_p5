"""

    Wenqing Fan
    Hang Yin
    cs5330 24sp project 5

    task 1 f Test the network on new inputs

    We followed the tutorial on https://nextjournal.com/gkoehler/pytorch-mnist
    for most of the process, and we organize the code into functions for further use

"""


import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

from p5_task1abcd import Net
    
# custom Image dataset to make the iterator to load the data
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [name for name in os.listdir(
            img_dir) if name.endswith('.png')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

# To plot the predictions for new inputs
def show_pred_inputs(custom_dataset, network):
    # Make predictions for each image
    fig = plt.figure()
    for i in range(10):
        image = custom_dataset[i]
        # print(f"image is :{image}")

        with torch.no_grad():
            output = network(image)
            print(f"output{i} is : {output}")
            prediction = output.data.max(1, keepdim=True)[1]

        plt.subplot(2, 5, i+1)

        plt.imshow(image[0], cmap='gray')
        plt.title(f"Prediction: {prediction[0][0]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# main function to load the state of the network
# and show the predicted for the hand written digits
def main(argv):
    # Load network model
    network = Net()
    network.load_state_dict(torch.load('/Users/hangyin/results/model.pth'))
    network.eval()

    # Define the location of images
    data_dir = '/Users/hangyin/Desktop/cs5330_24spring/project5/data/task1f_drawing'


    # Define transformations to apply to the images
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    # Load a custom dataset
    custom_dataset = CustomImageDataset(
        img_dir=data_dir,
        transform=transform
    )

    
    show_pred_inputs(custom_dataset, network)
    return


if __name__ == "__main__":
    import sys
    main(sys.argv)
