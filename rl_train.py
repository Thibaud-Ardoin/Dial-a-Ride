import argparse
import time
import sys
import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import namedtuple
from itertools import count

from utils import get_device, label2heatmap, visualize, indice_map2image, indice2image_coordonates, indices2image
from instances import PixelInstance
from models import NNPixelDataset
from models import UpAE, CNN1, CNN2, CNN3, UpCNN1, SeqFC1, NoPoolCNN1, SkipCNN1, CoCNN1, FC1, FC2, DQN
from transformer_model import Trans1
from rl_environment import DarEnv


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


class ReplayMemory(object):

    def __init__(self, capacity, transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transi = transition

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transi(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class RLTrainer():

    def __init__(self, flags, sacred=None):
        ''' Inintialisation of the trainner:
                Entends to load all the correct set up, ready to train
        '''

        # Incorporate arguments to the object parameters
        for key in flags:
            setattr(self, key, flags[key])

        self.sacred = sacred

        # Create saving experient dir
        if self.sacred :
            self.path_name = '/'.join([self.sacred.experiment_info['base_dir'], self.file_dir, str(self.sacred._id)])
        else :
            self.path_name = './data/experiments/' + self.alias + time.strftime("%d-%H-%M")
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
                json.dump(vars(self), f)

        self.device = get_device()

        if self.input_type:
            self.transform = transforms.Compose([])
        else :
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.ToTensor()
            ])

        # What would be of the ?
        self.GAMMA = 0.999
        self.eps_start = 0.5 #0.9
        self.eps_end = 0.05
        self.eps_decay = 5000 #200
        self.model_update = 10
        self.step = 0

        # RL elements
        self.env = DarEnv(size=self.image_size,
                          target_population=self.nb_target,
                          driver_population=self.nb_drivers)

        self.transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))

        self.memory = ReplayMemory(10000, self.transition)

        # Define NN
        try :
            if self.model=='FC1':
                self.model = globals()[self.model](self.image_size, self.layers).to(self.device)
            elif self.model=='FC2':
                self.model = globals()[self.model](self.image_size).to(self.device)
            elif self.model=='SeqFC1':
                self.model = globals()[self.model](4).to(self.device)
            elif self.model=='UpCNN1':
                self.model = globals()[self.model](2).to(self.device)
            elif self.model=='UpAE':
                self.model = globals()[self.model](self.image_size,
                                                         self.upscale_factor,
                                                         self.layers,
                                                         self.channels).to(self.device)
            elif self.model=='Trans1':
                self.model = globals()[self.model](src_vocab_size=self.image_size**2+1,
                                                         trg_vocab_size=self.image_size**2+1,
                                                         max_length=self.nb_target+1,
                                                         src_pad_idx=self.image_size,
                                                         trg_pad_idx=self.image_size,
                                                         dropout=self.dropout,
                                                         device=self.device).to(self.device)
            elif self.model=='DQN':
                self.model = globals()[self.model](size=self.image_size,
                                                   layer_size=self.layers).to(self.device)
            else :
                self.model = globals()[self.model]().to(self.device)
        except:
            raise "The model name has not been found !"

        # loss
        if self.criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif self.criterion == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        else :
            raise "Not found criterion"

        # optimizer
        if self.optimizer == 'Adam':
            self.opti = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.opti = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.95)
        elif self.optimizer == 'RMSprop' :
            self.opti = optim.RMSprop(policy_net.parameters())
        else :
            raise "Not found optimizer"

        # Scheduler
        if self.scheduler == 'plateau' :
            self.scheduler = ReduceLROnPlateau(self.opti, mode='min', patience=self.patience, factor=self.gamma)
        elif self.scheduler == 'step':
            self.scheduler = MultiStepLR(self.opti, milestones=self.milestones, gamma=self.gamma)

        # Loading model from dir
        if self.checkpoint_dir :
            self.model.load_state_dict(torch.load(self.checkpoint_dir + '/best_model.pt'), strict=False)
                #'./data/experiments/' + self.checkpoint_dir + '/best_model.pt'))

        # number of elements passed throgh the model for each epoch
        self.testing_size = self.batch_size * (10000 // self.batch_size)    #About 10k
        self.training_size = self.batch_size * (100000 // self.batch_size)   #About 100k

        self.statistics = {
            'reward': [],
            'duration': [],
            'accuracy': [],
            'loss': [],
            'epsylon': []
        }

        print(' *// What is this train about //* ')
        for item in vars(self):
            print(item, ':', vars(self)[item])



    def save_model(self, epoch):
        if self.checkpoint_type == 'best':
            name = 'best_model.pt'
        else : name = 'model_t=' + time.strftime("%d-%H-%M") + '_e=' + str(epoch) + '.pt'

        torch.save(self.model.state_dict(), '/'.join([self.path_name,name]))
        print(' - Done with saving ! - ')


    def plot_statistics(self, epoch, verbose=True, show=False):
        # Print them
        if verbose:
            print('\t ->[Epoch %d]<- loss: %.3f' % (epoch + 1, self.statistics['loss'][-1]))
            print('\t * Accuracy : %0.3f %%' % (self.statistics['accuracy'][-1]))
            print('\t * Reward : %0.3f' % (self.statistics['reward'][-1]))

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
        inputs, caracteristics = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
        # labels = neighbors[:,0]
        # shuffled_indexes = torch.randperm(neighbors.shape[1])
        # anonym_neighbors = neighbors[:,shuffled_indexes].to(self.device)

        if self.input_type=='map':
            outputs = self.model(inputs)

        elif self.input_type=='flatmap':
            target = torch.tensor([[self.image_size**2] for _ in range(inputs.shape[0])]).to(self.device).type(torch.LongTensor)
            outputs = self.model(inputs.to(self.device).type(torch.LongTensor),
                            target,
                            caracteristics)

        elif self.input_type=='map+coord':
            outputs = self.model(inputs, anonym_neighbors)
        elif self.input_type=='coord':
            outputs = self.model(anonym_neighbors)

        if self.output_type=='map':
            labels = label2heatmap(labels, self.image_size).to(self.device)
            labels = torch.argmax(labels, 1)
        elif self.output_type=='flatmap':
            labels = label2heatmap(labels, self.image_size).to(self.device)
            labels = torch.argmax(labels, 1)
            outputs = torch.squeeze(outputs[:, :, :-1]) #Remove inexistant  [-1] for start
        else :
            labels = labels.float()

        return outputs, labels

    def compile_stats(self, labels, outputs, loss, data):
        if self.output_type=='coord':
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

        elif self.output_type in ['map', 'flatmap']:
            predictions = torch.argmax(outputs, 1)
            correct = (predictions == labels).float().sum()
            # TODO : better metrics then 0 would be welcome !!
            nearest_accuracy = 0
            pointing_accuracy = 0

        return correct, nearest_accuracy, pointing_accuracy

    def save_visuals(self, epoch, data, outputs, labels, txt='test'):
        ''' Saving some examples of input -> output to see how the model behave '''
        print(' - Saving some examples - ')
        number_i = min(self.batch_size, 10)
        # print('\t \t + epoch::', epoch)
        # print('\t \t + data:', data[0].shape, data[0][:number_i])
        # print('\t \t + outputs:', outputs.shape, outputs[:number_i])
        # print('\t \t + labels:', labels.shape, labels[:number_i])
        plt.figure()
        fig, axis = plt.subplots(number_i, 2, figsize=(10, 50)) #2 rows for input, output
        fig.tight_layout()
        fig.suptitle(' - examples of network - ')
        for i in range(min(self.batch_size, number_i)):
            input_map = indices2image(data[0][i], self.image_size)
            axis[i, 0].imshow(input_map)
            im = indice_map2image(outputs[i], self.image_size).cpu().numpy()
            normalized = (im - im.min() ) / (im.max() - im.min())
            axis[i, 1].imshow(normalized)
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

        for epoch in range(self.epochs):
            running_loss = 0
            total = 0
            correct = nearest_accuracy = pointing_accuracy = 0

            self.model.train()
            for i, data in enumerate(trainloader):
                # set the parameter gradients to zero
                self.opti.zero_grad()

                outputs, labels = self.forward_data(data)
                loss = self.criterion(outputs, labels)

                loss.backward()
                # update the gradients
                self.opti.step()
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
                if self.checkpoint_type == 'best':
                    self.save_model(epoch=epoch)
            elif self.checkpoint_type == 'all':
                self.save_model(epoch=epoch)

        print(' - Done Training - ')


    def select_action(self, observation):
        sample = np.random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.step / self.eps_decay)
        self.step += 1
        if self.step > 1000 :
            eps_threshold = 0.1
        else :
            eps_threshold = 0.9
        self.statistics['epsylon'].append(eps_threshold)

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                observation = np.ascontiguousarray(observation, dtype=np.float32) / 255
                state = self.transform(torch.from_numpy(observation)).unsqueeze(0).to(self.device)

                return self.model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(self.env.action_space.n)]], device=self.device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        print(batch.state)
        print(batch.state.shape)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        print(state_batch)
        print(state_batch.shape)
        print(self.model(state_batch))
        print(self.model(state_batch).shape)


        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        #
        # print('state_action_values', state_action_values)
        # print(state_action_values.shape)
        # print('expected_state_action_values.unsqueeze(1)', expected_state_action_values.unsqueeze(1))
        # print(expected_state_action_values.unsqueeze(1).shape)

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        #torch.function.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        opti.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        opti.step()


    def train_rl(self):

        for i_episode in range(self.epochs):
            # Initialize the environment and state
            obs = self.env.reset()
            for t in count():

                # Select and perform an action
                action = self.select_action(obs)
                next_obs, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                obs = np.ascontiguousarray(obs.cpu(), dtype=np.float32) / 255
                obs = self.transform(torch.from_numpy(obs)).unsqueeze(0).to(self.device)
                next = np.ascontiguousarray(next_obs, dtype=np.float32) / 255
                next = self.transform(torch.from_numpy(next)).unsqueeze(0).to(self.device)

                # Store the transition in memory
                self.memory.push(obs, action, next, reward)

                # Move to the next state
                obs = next_obs

                # Perform one step of the optimization (on the target network)
                self.optimize_model()

                if done:
                    self.statistics['duration'].append(t + 1)
                    self.statistics['reward'].append(self.env.cumulative_reward)
                    self.statistics['loss'].append(0)
                    self.statistics['accuracy'].append(0)
                    self.plot_statistics(i_episode)
                    break

            # # Update the target network, copying all weights and biases in DQN
            # if i_episode % self.model_update == 0:
            #     target_net.load_state_dict(policy_net.state_dict())


    def run(self):
        ''' Getting the rl training to go'''

        # Get number of actions from gym action space

        n_actions = self.env.action_space.n
        print('Sizie of the action space: ', n_actions)

        # policy_net = DQN(size=IMAGE_SIZE, upscale_factor=2, layer_size=128, channels=1).to(device)
        # target_net = DQN(size=IMAGE_SIZE, upscale_factor=2, layer_size=128, channels=1).to(device)
        # target_net.load_state_dict(policy_net.state_dict())
        # target_net.eval()

        # opti = optim.RMSprop(policy_net.parameters())

        self.train_rl()

if __name__ == '__main__':

    # Get params
    parameters = parse_args(sys.argv[1:])

    # Get the trainer object
    trainer = RLTrainer(parameters)

    # Start a train
    trainer.run()
