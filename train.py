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
from tqdm import tqdm

from utils import get_device
from generator import PixelInstance
from models import NNPixelDataset, DataLoader, CNN1, CNN2, CNN3, SkipCNN1, CoCNN1, FC1, NoPoolCNN1, CoCNNNoPool1


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a train.",
        epilog="python train.py --epochs INT")

    # required input parameters
    parser.add_argument(
        '--epochs', type=int,  default=1,
        help='Number of Epochs the train is going to last for !')

    parser.add_argument(
        '--alias', type=str,  default='testing',
        help='Nickname you give to this experimentation range')

    parser.add_argument(
        '--lr', type=float,  default=0.001,
        help='Learning Rate for training aim')

    parser.add_argument(
        '--data', type=str, default='./data/instances/split3_50k_n2_s50/',
        help='Directory of the 3_splited  data')

    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size')

    parser.add_argument(
        '--shuffle', type=bool, default=True,
        help='Should the data be shuffled when used in the data loader ?')

    parser.add_argument(
        '--optimizer', type=str, default='Adam', choices=['Adam'],
        help='optimizer used for the training process')

    parser.add_argument(
        '--criterion', type=str, default='MSE', choices=['MSE', 'l1'],
        help='How should the loss be calculated')

    parser.add_argument(
        '--model', type=str, default='CNN1',
        help='Model defined in models.py that should be used :). For ex: CNN1, FC1, ect.')

    parser.add_argument(
        '--checkpoint_type', type=str, default='best', choices=['all', 'best', 'none'],
        help='Model defined in models.py that should be used :)')

    parser.add_argument(
        '--milestones', nargs='+', type=int, default=[50],
        help='List of milestones needed to decay the LR by a factor of gamma')

    parser.add_argument(
        '--gamma', type=float, default=0.1,
        help='Decay factor for the learning rate')

    parser.add_argument(
        '--patience', type=int, default=5,
        help='Number of step without decreasing of the loss before \
        reducing the learning rate')

    parser.add_argument(
        '--scheduler', type=str, default='plateau', choices=['plateau', 'step'],
        help='Type of learning rate scheduer')

    parser.add_argument(
        '--checkpoint_dir', type=str, default='',
        help='Directory for loading the checkpoint')

    parser.add_argument(
        '--input_type', type=str, default='map',
        help='Type of the data input in the model')

    return parser.parse_known_args(args)[0]

#######
# Epoch wise testing process
#######

def testing(model, testloader, criterion, testing_size, input_type):
    loss = 0
    correct = 0
    pointing_accuracy = 0
    nearest_accuracy = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, neighbors = data[0].to(device, non_blocking=True), data[1]
            labels = neighbors[:,0].to(device, non_blocking=True)

            shuffled_indexes = torch.randperm(neighbors.shape[1])
            anonym_neighbors = neighbors[:,shuffled_indexes].to(device)

            if input_type=='map':
                outputs = model(inputs)
            elif input_type=='map+coord':
                outputs = model(inputs, anonym_neighbors)

            loss += criterion(outputs, labels.float())
            total += labels.size(0)
            rounded = torch.round(outputs)
            rx, ry = rounded.reshape(2, labels.size(0))
            lx, ly = labels.reshape(2, labels.size(0))
            correct += ((rx == lx) & (ry == ly)).sum().item()

            distance_pred2points = list(map(lambda x: np.linalg.norm(rounded.cpu() - x, axis=1), neighbors.cpu()[0]))
            # Case where the model aims perfectly to one pixel
            pointing_accuracy += (np.sum(np.min(distance_pred2points, axis=0) == 0))
            # Case where the nearest pixel to prediction is the nearest_neighbors
            nearest_accuracy += np.sum(np.argmin(distance_pred2points, axis=0) == 0)

            if (i >= testing_size): break
    return {'loss': loss / total,
            'accuracy': 100 * correct / total,
            'nearest_acc': 100 * nearest_accuracy / total,
            'point_acc': 100 * pointing_accuracy / total  }

#######
# Training function
######

def train(model, trainloader, testloader,  number_epochs, criterion, optimizer, scheduler, testing_size, name, checkpoint_type, input_type):
    print(' - Start Training - ')
    max_test_accuracy = 0
    training_statistics = {
        'test_accuracy': [],
        'train_loss': [],
        'test_loss': [],
        'test_nearest_acc':[],
        'test_point_acc':[]
    }
    for epoch in range(number_epochs):
        model.train()
        running_loss = 0
        total = 0

        for data in tqdm(trainloader):
            if input_type=='map':
                # data pixels and labels to GPU if available
                inputs, neighbors = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
                labels = neighbors[:,0]
                # set the parameter gradients to zero
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                # propagate the loss backward

            elif input_type=='map+coord':
                inputs, neighbors = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
                labels = neighbors[:,0]

                shuffled_indexes = torch.randperm(neighbors.shape[1])
                anonym_neighbors = neighbors[:,shuffled_indexes].to(device)

                optimizer.zero_grad()
                outputs = model(inputs, anonym_neighbors)

                loss = criterion(outputs, labels.float())

            loss.backward()
            # update the gradients
            optimizer.step()
            total += labels.size(0)
            running_loss += loss.item()

        scheduler.step(running_loss)
        # Print results for epoch
        print('\t * [Epoch %d] loss: %.3f' %
                      (epoch + 1, running_loss / total))
        # print('\t * Scheduler, the current learning rate is: ', scheduler.get_lr())
        test_statistics = testing(model, testloader, criterion, testing_size, input_type)
        print('\t * Testing accuracy : %0.3f %%' % (test_statistics['accuracy']))
        print('\t * Testing loss : %0.3f' % (test_statistics['loss']))
        print('\t * Testing right pointing accuracy : %0.3f' % (test_statistics['point_acc']))
        print('\t * Testing nearest accuracy : %0.3f' % (test_statistics['nearest_acc']))

        # Compile results
        training_statistics['test_accuracy'].append(test_statistics['accuracy'])
        training_statistics['test_loss'].append(test_statistics['loss'])
        training_statistics['test_nearest_acc'].append(test_statistics['nearest_acc'])
        training_statistics['test_point_acc'].append(test_statistics['point_acc'])
        training_statistics['train_loss'].append(running_loss)
        plot_statistics(training_statistics, name)

        #
        if test_statistics['accuracy'] > max_test_accuracy :
            max_test_accuracy = test_statistics['accuracy']
            if checkpoint_type == 'best':
                save_model(model, path_name=name, checkpoint=checkpoint_type, epoch=epoch)
        elif checkpoint_type == 'all':
            save_model(model, path_name=name, checkpoint=checkpoint_type, epoch=epoch)

    print(' - Done Training - ')
    return training_statistics

######
# Final validation step
######

def validation(model, validationLoader, criterion, input_type):
    print(' - Start Validation provcess - ')
    loss = 0
    correct = 0
    total = 0
    pointing_accuracy = 0
    nearest_accuracy = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(validationLoader):
            inputs, neighbors = data[0].to(device, non_blocking=True), data[1]
            labels = neighbors[:,0].to(device, non_blocking=True)

            shuffled_indexes = torch.randperm(neighbors.shape[1])
            anonym_neighbors = neighbors[:,shuffled_indexes].to(device)

            if input_type=='map':
                outputs = model(inputs)
            elif input_type=='map+coord':
                outputs = model(inputs, anonym_neighbors)

            loss += criterion(outputs, labels.float())
            total += labels.size(0)
            rounded = torch.round(outputs)
            rx, ry = rounded.reshape(2, labels.size(0))
            lx, ly = labels.reshape(2, labels.size(0))
            correct += ((rx == lx) & (ry == ly)).sum().item()

            distance_pred2points = list(map(lambda x: np.linalg.norm(rounded.cpu() - x, axis=1), neighbors.cpu()[0]))
            # Case where the model aims perfectly to one pixel
            # pointing_accuracy += (np.sum(np.min(distance_pred2points, axis=0) == 0))
            # Case where the nearest pixel to prediction is the nearest_neighbors
            nearest_accuracy += np.sum(np.argmin(distance_pred2points, axis=0) == 0)
        print('\t * Validation run -- ' )
        print('\t - Validation accuracy : %0.3f %%' % (correct / total))
        print('\t - Validation loss : %0.3f' % (loss / total))
        print('\t - Validation right pointing accuracy : %0.3f' % (0))
        print('\t - Validation nearest accuracy : %0.3f' % (nearest_accuracy / total))


def plot_statistics(statistics, name, show=False):
    # Create plot of the statiscs, saved in folder
    colors = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple']
    fig, (axis) = plt.subplots(1, len(statistics), figsize=(20, 10))
    fig.suptitle(' - Training: ' + name)
    for i, key in enumerate(statistics):
        axis[i].plot(statistics[key], colors[i])
        axis[i].set_title(' Plot of ' + key)
    if show :
        fig.show()
    fig.savefig(path_name + '/result_figure.png')
    fig.clf()
    plt.close(fig)

    # Save the statistics as CSV file
    try:
        with open(path_name + '/statistics.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=statistics.keys())
            writer.writeheader()
            # for key in statistics
            writer.writerow(statistics)
    except IOError:
        print("I/O error")


def save_model(model, path_name, checkpoint, epoch):
    if checkpoint == 'best':
        name = 'best_model.pt'
    else : name = 'model_t=' + time.strftime("%d-%H-%M") + '_e=' + str(epoch) + '.pt'
    torch.save(model.state_dict(), '/'.join([path_name,name]))
    print(' - Done with saving ! - ')


if __name__ == '__main__':
    flags = parse_args(sys.argv[1:])

    # Create saving experient dir
    path_name = './data/experiments/' + flags.alias + time.strftime("%d-%H-%M")
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    else :
        print(' ! OverWriting ! ')

    # Save parameters
    with open(path_name + '/parameters.json', 'w') as f:
        json.dump(vars(flags), f)

    device = get_device()
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])
    ])


    # Define Datasets
    train_data = NNPixelDataset(flags.data + '/train_instances.pkl', transform)
    test_data = NNPixelDataset(flags.data + '/test_instances.pkl', transform)

    # Define dataloaders
    trainloader = DataLoader(train_data, batch_size=flags.batch_size, shuffle=flags.shuffle)
    testloader = DataLoader(test_data, batch_size=flags.batch_size, shuffle=flags.shuffle)
    print(' - Done with loading the data - ')


    # Define NN
    try :
        if flags.model=='FC1':
            model = locals()[flags.model](50, 128).to(device)
        else :
            model = locals()[flags.model]().to(device)
    except:
        raise "The model as input has not been found !"
    # if flags.model == 'CNN2':
    #     model = CNN2().to(device)
    # elif flags.model == 'CNN1':
    #     model = CNN1().to(device)
    # elif flags.model == 'CNN3':
    #     model = CNN3().to(device)
    # elif flags.model == 'Skip_CNN1':
    #     model = Skip_CNN1().to(device)
    # else:
    #     raise "Not found model"

    print(' - Network: ', model)

    # loss
    if flags.criterion == 'MSE':
        criterion = nn.MSELoss()
    elif flags.criterion == 'l1':
        criterion = nn.L1Loss()
    else :
        raise "Not found criterion"

    # optimizer
    if flags.optimizer == 'Adam':
         optimizer = optim.Adam(model.parameters(), lr=flags.lr)
    else :
        raise "Not found optimizer"

    # Scheduler
    if flags.scheduler == 'plateau' :
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=flags.patience, factor=flags.gamma)
    elif flags.scheduler == 'step':
        scheduler = MultiStepLR(optimizer, milestones=flags.milestones, gamma=flags.gamma)

    # Loading model from dir
    if flags.checkpoint_dir :
        model.load_state_dict(torch.load('./data/experiments/' + flags.checkpoint_dir + '/best_model.pt'))

    # Start training and testing
    traing_statistics = train(model=model,
                              trainloader=trainloader,
                              testloader=testloader,
                              number_epochs=flags.epochs,
                              criterion=criterion,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              testing_size=flags.batch_size*4,
                              name=path_name,
                              checkpoint_type=flags.checkpoint_type,
                              input_type=flags.input_type)

    plot_statistics(traing_statistics, path_name)

    # free some memory
    del train_data
    del trainloader
    del test_data
    del testloader

    # Validation run
    validation_data = NNPixelDataset(flags.data + '/validation_instances.pkl', transform)
    validationLoader = DataLoader(validation_data, batch_size=flags.batch_size, shuffle=flags.shuffle)
    validation(model, validationLoader, criterion, flags.input_type)
    print(' - Done with Training - ')
