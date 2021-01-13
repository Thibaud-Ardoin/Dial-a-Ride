import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim

from instances import PixelInstance
from utils import get_device, visualize


class NNPixelDataset(Dataset):
    """ Customed Dataset class for our Instances data
    """
    def __init__(self, data_path, transforms, channels):
        filehandler = open(data_path, 'rb')
        self.data = pickle.load(filehandler)
        filehandler.close()
        self.channels = channels

        self.transforms = transforms


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        """ Returns a couple (image, neares 1hot)"""
        instance = self.data[idx]

        # visualize(instance.image, txt='From instance data directly')

        if self.channels==1:
            image = np.asarray(instance.image).reshape(instance.size, instance.size) #
        else :
            image = np.asarray(instance.image).reshape(instance.size, instance.size, self.channels)

        # visualize(image, txt='From Pixel dataset before transforms')
        image = image.astype(np.uint8)
        image = self.transforms(image)

        # visualize(image, txt='From Pixel dataset output')

        return (image, torch.tensor(instance.neighbor_list))


class CNN1(nn.Module):
    """ Neural network definition
    """
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=7744, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN2(nn.Module):
    """ Neural network definition
    """
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32,
                               kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=32, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN3(nn.Module):
    """ Neural network definition
    """
    def __init__(self):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=7744, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class UpCNN1(nn.Module):
    """ Neural network definition
    """
    def __init__(self, upscale_factor):
        super(UpCNN1, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

        self.fc1 = nn.Linear(in_features=10000, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        nn.init.orthogonal_(self.conv1.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv4.weight)


class UpAE(nn.Module):
    def __init__(self, size, upscale_factor, layer_size, channels):
        super(UpAE, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

        self.fc1 = nn.Linear(in_features=(upscale_factor*size)**2, out_features=layer_size)
        self.fc2 = nn.Linear(in_features=layer_size, out_features=size**2)

    def forward(self, input_image):
        x = self.relu(self.conv1(input_image))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        upscaled_image = self.pixel_shuffle(self.conv4(x))
        image_vector = upscaled_image.view(upscaled_image.size(0), -1)
        x = self.relu(self.fc1(image_vector))
        reconstruction = self.fc2(x)
        return reconstruction

    def _initialize_weights(self):
        nn.init.orthogonal_(self.conv1.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv4.weight)


class NoPoolCNN1(nn.Module):
    """ Neural network definition
    """
    def __init__(self):
        super(NoPoolCNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16,
                               kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=33856, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        # x = F.avg_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SkipCNN1(nn.Module):
    """ Neural network definition
    """
    def __init__(self):
        super(SkipCNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=4608, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x1 = x
        x = F.relu(self.conv2(x) + x1)
        x = F.avg_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CoCNN1(nn.Module):
    """ Neural network definition:
        Add as entry the vehicles positions  as coordonates
        The output will then be a 1hot vector to target the right coordonate.
    """
    def __init__(self):
        super(CoCNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=3876, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)


    def forward(self, map, coord):
        x = map
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        flat_coord = coord.float().view(coord.size(0), -1)

        # print(flat_coord.size())
        # print(x.size())

        x = F.relu(self.fc1(torch.cat((x, flat_coord), 1)))
        x = self.fc2(x)
        return x

class CoCNNNoPool1(nn.Module):
    """ Neural network definition:
        Add as entry the vehicles positions  as coordonates
        The output will then be a 1hot vector to target the right coordonate.
    """
    def __init__(self):
        super(CoCNNNoPool1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=67716, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)


    def forward(self, map, coord):
        x = map
        x = F.relu(self.conv1(x))
        # x = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        # x = F.avg_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        flat_coord = coord.float().view(coord.size(0), -1)

        # print(flat_coord.size())
        # print(x.size())

        x = F.relu(self.fc1(torch.cat((x, flat_coord), 1)))
        x = self.fc2(x)
        return x

class FC1(nn.Module):
    """ Neural network definition
    """
    def __init__(self, size, hidden_layers):
        super(FC1, self).__init__()
        self.size = size
        self.hidden_layers = hidden_layers
        self.fc1 = nn.Linear(in_features=self.size**2, out_features=self.hidden_layers)
        self.fc2 = nn.Linear(in_features=self.hidden_layers, out_features=2)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FC2(nn.Module):
    """ Neural network definition
    """
    def __init__(self, size):
        super(FC2, self).__init__()
        self.size = size
        self.fc1 = nn.Linear(in_features=self.size**2, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)
        self.fc3 = nn.Linear(in_features=2, out_features=4)
        self.fc4 = nn.Linear(in_features=4, out_features=2)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class SeqFC1(nn.Module):
    """ Neural network definition
    """
    def __init__(self, size):
        super(SeqFC1, self).__init__()
        self.size = size
        self.fc1 = nn.Linear(in_features=self.size, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=2)


    def forward(self, coord):
        x = coord.float().view(coord.size(0), -1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':

    device = get_device()

    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor()
    ])

    # Define Datasets
    train_data = NNPixelDataset('./data/instances/split3_100k/train_instances.pkl', transform)
    test_data = NNPixelDataset('./data/instances/split3_100k/test_instances.pkl', transform)
    test_data = NNPixelDataset('./data/instances/split3_100k/validation_instances.pkl', transform)

    # Define dataloaders
    trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
    testloader = DataLoader(test_data, batch_size=128, shuffle=True)

    # Define NN
    net = Net().to(device)
    print(' - Network: ', net)

    # loss
    criterion = nn.MSELoss()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    number_epochs=20

    # Start training and testing
    train(net, trainloader, number_epochs, criterion, optimizer)
    test(net, testloader, criterion)

    print('Finished Training')

    # Save model
    PATH = './data/models/'
    name = 'model_t=' + time.strftime("%d-%H-%M") + '.pt'
    torch.save(net.state_dict(), PATH + name)
    print(' - Done with saving ! - ')
