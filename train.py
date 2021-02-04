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

from utils import get_device, label2heatmap, visualize, indice_map2image, indice2image_coordonates, indices2image
from instances import PixelInstance
from models import NNPixelDataset
from models import UpAE, CNN1, CNN2, CNN3, UpCNN1, SeqFC1, NoPoolCNN1, SkipCNN1, CoCNN1, FC1, FC2
from transformer_model import Trans1


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
        '--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'],
        help='optimizer used for the training process')

    parser.add_argument(
        '--criterion', type=str, default='MSE', choices=['MSE', 'l1', 'crossentropy'],
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

    parser.add_argument(
        '--output_type', type=str, default='coord',
        help='Type of the data output in the model')

    parser.add_argument(
        '--layers', type=int, default=128,
        help='If needed, this value gives the size of hidden layers')

    return parser.parse_known_args(args)[0]

#######
# Epoch wise testing process
#######



#######
# Training function
######


######
# Final validation step
######







class Trainer():

    def __init__(self, flags, sacred=None):
        ''' Inintialisation of the trainner:
                Entends to load all the correct set up, ready to train
        '''
        self.flags = flags
        self.sacred = sacred

        a, b, c, d, e, f, g = self.flags.data.split('_')
        self.unique_nn = int(b[0])
        self.data_number = int(c[:-1])
        self.population = int(d[1:])
        self.image_size = int(e[1:])
        self.moving_car = int(f[1:])
        self.indice_list = int(g[1:])
        if self.moving_car:
            self.channels = 2
        else:
            self.channels = 1

        # Create saving experient dir
        if self.sacred :
            self.path_name = '/'.join([self.sacred.experiment_info['base_dir'], self.flags.file_dir, str(self.sacred._id)])
        else :
            self.path_name = './data/experiments/' + self.flags.alias + time.strftime("%d-%H-%M")
            print(' ** Saving train path: ', self.path_name)
            if not os.path.exists(self.path_name):
                os.makedirs(self.path_name)
            else :
                print(' Already such a path.. adding random seed')
                self.path_name = self.path_name + '#' + np.randint(100, 1000)

        # Save parameters
        if self.sacred :
            pass
        else :
            with open(self.path_name + '/parameters.json', 'w') as f:
                json.dump(vars(self.flags), f)

        self.device = get_device()

        if self.indice_list:
            self.transform = transforms.Compose([])
        else :
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.ToTensor()
            ])

        # Define NN
        try :
            if self.flags.model=='FC1':
                self.model = globals()[self.flags.model](self.image_size, self.flags.layers).to(self.device)
            elif self.flags.model=='FC2':
                self.model = globals()[self.flags.model](self.image_size).to(self.device)
            elif self.flags.model=='SeqFC1':
                self.model = globals()[self.flags.model](4).to(self.device)
            elif self.flags.model=='UpCNN1':
                self.model = globals()[self.flags.model](2).to(self.device)
            elif self.flags.model=='UpAE':
                self.model = globals()[self.flags.model](self.image_size,
                                                         self.flags.upscale_factor,
                                                         self.flags.layers,
                                                         self.channels).to(self.device)
            elif self.flags.model=='Trans1':
                self.model = globals()[self.flags.model](src_vocab_size=self.image_size**2+1,
                                                         trg_vocab_size=self.image_size**2+1,
                                                         max_length=self.population+1,
                                                         src_pad_idx=self.image_size,
                                                         trg_pad_idx=self.image_size,
                                                         dropout=self.flags.dropout,
                                                         device=self.device).to(self.device)
            else :
                self.model = globals()[self.flags.model]().to(self.device)
        except:
            raise "The model name has not been found !"

        # loss
        if self.flags.criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.flags.criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif self.flags.criterion == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        else :
            raise "Not found criterion"

        # optimizer
        if self.flags.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.flags.lr)
        elif self.flags.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.flags.lr, momentum=0.95)
        else :
            raise "Not found optimizer"

        # Scheduler
        if self.flags.scheduler == 'plateau' :
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.flags.patience, factor=self.flags.gamma)
        elif self.flags.scheduler == 'step':
            self.scheduler = MultiStepLR(self.optimizer, milestones=self.flags.milestones, gamma=self.flags.gamma)

        # Loading model from dir
        if self.flags.checkpoint_dir :
            self.model.load_state_dict(torch.load(self.flags.checkpoint_dir + '/best_model.pt'), strict=False)
                #'./data/experiments/' + self.flags.checkpoint_dir + '/best_model.pt'))

        # number of elements passed throgh the model for each epoch
        self.testing_size = min(self.flags.batch_size * (10000 // self.flags.batch_size), self.data_number)     #About 10k
        self.training_size = min(self.flags.batch_size * (100000 // self.flags.batch_size), self.data_number)   #About 100k

        self.statistics = {
            'train_accuracy': [],
            'train_loss': [],
            'train_pointing_acc': [],
            'train_nearest_acc': [],
            'test_accuracy': [],
            'test_loss': [],
            'test_nearest_acc':[],
            'test_pointing_acc':[]
        }

        print(' *// What is this train about //* ')
        for item in vars(self):
            print(item, ':', vars(self)[item])



    def save_model(self, epoch):
        if self.flags.checkpoint_type == 'best':
            name = 'best_model.pt'
        else : name = 'model_t=' + time.strftime("%d-%H-%M") + '_e=' + str(epoch) + '.pt'

        torch.save(self.model.state_dict(), '/'.join([self.path_name,name]))
        print(' - Done with saving ! - ')


    def plot_statistics(self, epoch, verbose=True, show=False):
        # Print them
        if verbose:
            print('\t ->[Epoch %d]<- loss: %.3f' % (epoch + 1, self.statistics['train_loss'][-1]))
            print('\t * Testing accuracy : %0.3f %%' % (self.statistics['test_accuracy'][-1]))
            print('\t * Testing loss : %0.3f' % (self.statistics['test_loss'][-1]))

        # Create plot of the statiscs, saved in folder
        colors = [plt.cm.tab20(0),plt.cm.tab20(1),plt.cm.tab20c(2),
                  plt.cm.tab20c(3), plt.cm.tab20c(4),
                  plt.cm.tab20c(5),plt.cm.tab20c(6),plt.cm.tab20c(7)]
        fig, (axis) = plt.subplots(1, len(self.statistics), figsize=(20, 10))
        fig.suptitle(' - Training: ' + self.path_name)
        for i, key in enumerate(self.statistics):
            # Sacred (The one thing to keep here)
            if self.sacred :
                self.sacred.log_scalar(key, self.statistics[key][-1], len(self.statistics[key]))
            axis[i].plot(self.statistics[key], color=colors[i])
            axis[i].set_title(' Plot of ' + key)
        if show :
            fig.show()
        fig.savefig(self.path_name + '/result_figure.png')
        fig.clf()
        plt.close(fig)

        # Save the statistics as CSV file
        if not self.sacred:
            try:
                with open(self.path_name + '/statistics.csv', 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=statistics.keys())
                    writer.writeheader()
                    # for key in statistics
                    writer.writerow(self.statistics)
            except IOError:
                print("I/O error")


    def forward_data(self, data):
        inputs, neighbors = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
        labels = neighbors[:,0]
        shuffled_indexes = torch.randperm(neighbors.shape[1])
        anonym_neighbors = neighbors[:,shuffled_indexes].to(self.device)

        if self.flags.input_type=='map':
            outputs = self.model(inputs)
        elif self.flags.input_type=='flatmap':
            outputs = self.model(inputs.to(self.device).type(torch.LongTensor),
                            torch.tensor([[self.image_size**2] for _ in range(inputs.shape[0])]).to(self.device).type(torch.LongTensor))
        elif self.flags.input_type=='map+coord':
            outputs = self.model(inputs, anonym_neighbors)
        elif self.flags.input_type=='coord':
            outputs = self.model(anonym_neighbors)

        if self.flags.output_type=='map':
            labels = label2heatmap(labels, self.image_size).to(self.device)
            labels = torch.argmax(labels, 1)
        elif self.flags.output_type=='flatmap':
            labels = label2heatmap(labels, self.image_size).to(self.device)
            labels = torch.argmax(labels, 1)
            outputs = torch.squeeze(outputs[:, :, :-1]) #Remove inexistant  [-1] for start
        else :
            labels = labels.float()

        return outputs, labels

    def compile_stats(self, labels, outputs, loss, data):
        if self.flags.output_type=='coord':
            rounded = torch.round(outputs)
            rx, ry = rounded.reshape(2, labels.size(0))
            lx, ly = labels.reshape(2, labels.size(0))
            correct = ((rx == lx) & (ry == ly)).sum().item()

            distance_pred2points = list(map(lambda x: np.linalg.norm(rounded.cpu() - x, axis=1),
                                            data[1].to(self.device, non_blocking=True).cpu()[0]))
            # Case where the model aims perfectly to one pixel
            pointing_accuracy = (np.sum(np.min(distance_pred2points, axis=0) == 0))
            # Case where the nearest pixel to prediction is the nearest_neighbors
            nearest_accuracy = np.sum(np.argmin(distance_pred2points, axis=0) == 0)

        elif self.flags.output_type in ['map', 'flatmap']:
            predictions = torch.argmax(outputs, 1)
            correct = (predictions == labels).float().sum()
            # TODO : better metrics then 0 would be welcome !!
            nearest_accuracy = 0
            pointing_accuracy = 0

        return correct, nearest_accuracy, pointing_accuracy

    def save_visuals(self, epoch, data, outputs, labels, txt='test'):
        ''' Saving some examples of input -> output to see how the model behave '''
        print(' - Saving some examples - ')
        number_i = min(self.flags.batch_size, 10)
        # print('\t \t + epoch::', epoch)
        # print('\t \t + data:', data[0].shape, data[0][:number_i])
        # print('\t \t + outputs:', outputs.shape, outputs[:number_i])
        # print('\t \t + labels:', labels.shape, labels[:number_i])
        plt.figure()
        fig, axis = plt.subplots(2, number_i, figsize=(50, 25)) #3 rows for input, output, processed
        fig.tight_layout()
        fig.suptitle(' - examples of network - ')
        for i in range(min(self.flags.batch_size, number_i)):
            input_map = indices2image(data[0][i], self.image_size)
            axis[0, i].imshow(input_map)
            im = indice_map2image(outputs[i], self.image_size).cpu().numpy()
            normalized = (im - im.min() ) / (im.max() - im.min())
            axis[1, i].imshow(normalized)
        img_name = self.path_name + '/example_epoch' + str(epoch) + '.png'
        plt.savefig(img_name)
        plt.close()
        if self.sacred :
            self.sacred.add_artifact(img_name, content_type='image')


    def testing(self, testloader, epoch):
        loss = 0
        total = 0
        correct = nearest_accuracy = pointing_accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader):

                outputs, labels = self.forward_data(data)
                loss += self.criterion(outputs, labels)
                total += labels.size(0)

                c, n, p = self.compile_stats(labels, outputs, loss, data)
                correct += c ;  pointing_accuracy += p ; nearest_accuracy += n

                if (i >= self.testing_size): break
        self.save_visuals(epoch, data, outputs, labels)
        self.statistics['test_accuracy'].append(100*(correct / total).cpu().item())
        self.statistics['test_loss'].append((loss / total).cpu().item())
        self.statistics['test_nearest_acc'].append(100*(nearest_accuracy / total))
        self.statistics['test_pointing_acc'].append(100*(pointing_accuracy / total))


    def train(self, trainloader, testloader):
        print(' - Start Training - ')
        max_test_accuracy = 0

        for epoch in range(self.flags.epochs):
            running_loss = 0
            total = 0
            correct = nearest_accuracy = pointing_accuracy = 0

            self.model.train()
            for i, data in enumerate(trainloader):
                # set the parameter gradients to zero
                self.optimizer.zero_grad()

                outputs, labels = self.forward_data(data)
                loss = self.criterion(outputs, labels)

                loss.backward()
                # update the gradients
                self.optimizer.step()
                total += labels.size(0)
                running_loss += loss.item()

                # Compile statistics
                c, n, p = self.compile_stats(labels, outputs, loss, data)
                correct += c ;  pointing_accuracy += p ; nearest_accuracy += n

                if (i >= self.training_size): break

            # Compile results
            self.statistics['train_loss'].append(running_loss / total)
            self.statistics['train_accuracy'].append(100*(correct / total).cpu().item())
            self.statistics['train_pointing_acc'].append(100*(pointing_accuracy / total))
            self.statistics['train_nearest_acc'].append(100*(nearest_accuracy / total))

            # Start Testings
            self.testing(testloader, epoch)

            # Stats tratment and update
            self.plot_statistics(epoch)
            self.scheduler.step(running_loss)

            if self.statistics['test_accuracy'][-1] == np.max(self.statistics['test_accuracy']) :
                if self.flags.checkpoint_type == 'best':
                    self.save_model(epoch=epoch)
            elif self.flags.checkpoint_type == 'all':
                self.save_model(epoch=epoch)

        print(' - Done Training - ')

    def validation(self, validationLoader):
        print(' - Start Validation provcess - ')
        loss = 0
        total = 0
        correct = pointing_accuracy = nearest_accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(validationLoader):
                outputs, labels = self.forward_data(data)
                loss += self.criterion(outputs, labels)
                total += labels.size(0)

                c, n, p = self.compile_stats(labels, outputs, loss, data)
                correct += c ;  pointing_accuracy += p ; nearest_accuracy += n

            print('\t * Validation run -- ' )
            print('\t - Validation accuracy : %0.3f %%' % (100 * correct / total))
            print('\t - Validation loss : %0.3f' % (loss / total))
            print('\t - Validation right pointing accuracy : %0.3f' % (100 *pointing_accuracy / total))
            print('\t - Validation nearest accuracy : %0.3f' % (100 * nearest_accuracy / total))


    def run(self):
        ''' Loading the data and starting the training '''

        # Define Datasets
        train_data = NNPixelDataset(self.flags.data + '/train_instances.pkl', self.transform, channels=self.channels, isList=self.indice_list)
        test_data = NNPixelDataset(self.flags.data + '/test_instances.pkl', self.transform, channels=self.channels, isList=self.indice_list)

        # Define dataloaders
        trainloader = DataLoader(train_data, batch_size=self.flags.batch_size, shuffle=self.flags.shuffle)
        testloader = DataLoader(test_data, batch_size=self.flags.batch_size, shuffle=self.flags.shuffle)
        print(' - Done with loading the data - ')

        # Start training and testing
        self.train(trainloader=trainloader,
                   testloader=testloader)

        # free some memory
        del train_data
        del trainloader
        del test_data
        del testloader


    def evaluation(self):
        ''' Evaluation process of the Trainer
        '''
        validation_data = NNPixelDataset(self.flags.data + '/validation_instances.pkl', self.transform, channels=self.channels, isList=self.indice_list)
        validationLoader = DataLoader(validation_data, batch_size=self.flags.batch_size, shuffle=self.flags.shuffle)
        self.validation(validationLoader)

        del validation_data
        del validationLoader

        print(' - Done with Training - ')


if __name__ == '__main__':

    # Get params
    parameters = parse_args(sys.argv[1:])

    # Get the trainer object
    trainer = Trainer(parameters)

    # Start a train
    trainer.run()

    # Start evaluation
    trainer.evaluation()
