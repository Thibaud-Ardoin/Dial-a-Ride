import argparse
import time
import sys
import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_device, label2heatmap
from generator import PixelInstance
from models import NNPixelDataset
from models import UpAE, CNN1, CNN2, CNN3, UpCNN1, SeqFC1, NoPoolCNN1, SkipCNN1, CoCNN1, FC1, FC2


class Tester():

    def __init__(self, flags, saved_model):
        ''' Inintialisation of the trainner:
                Entends to load all the correct set up, ready to train
        '''
        self.flags = flags

        self.device = get_device()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor()
        ])

        # Define NN
        try :
            if self.flags.model=='FC1':
                self.model = globals()[self.flags.model](50, 128).to(self.device)
            elif self.flags.model=='FC2':
                self.model = globals()[self.flags.model](50).to(self.device)
            elif self.flags.model=='SeqFC1':
                self.model = globals()[self.flags.model](4).to(self.device)
            elif self.flags.model=='UpCNN1':
                self.model = globals()[self.flags.model](2).to(self.device)
            elif self.flags.model=='UpAE':
                self.model = globals()[self.flags.model](50, 2).to(self.device)
            else :
                self.model = globals()[self.flags.model]().to(self.device)
        except:
            raise "The model as input has not been found !"

        print(' - Network: ', self.model)

        # loss
        if self.flags.criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.flags.criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif self.flags.criterion == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        else :
            raise "Not found criterion"

        # Loading model from dir
        if saved_model :
            self.model.load_state_dict(torch.load(saved_model), strict=False)
                #'./data/experiments/' + self.flags.checkpoint_dir + '/best_model.pt'))

    def run(self):
        ''' Loading the data and starting the training '''

        # Define Datasets
        train_data = NNPixelDataset(self.flags.data + '/train_instances.pkl', self.transform)
        test_data = NNPixelDataset(self.flags.data + '/test_instances.pkl', self.transform)

        # Define dataloaders
        trainloader = DataLoader(train_data, batch_size=self.flags.batch_size, shuffle=self.flags.shuffle)
        testloader = DataLoader(test_data, batch_size=self.flags.batch_size, shuffle=self.flags.shuffle)
        print(' - Done with loading the data - ')




if __name__ == '__main__':

    # Get params
    parameters = parse_args(sys.argv[1:])

    # Get the tester object
    tester = Tester(parameters)
