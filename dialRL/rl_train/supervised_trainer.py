import gym
import logging
import os
import imageio
import time
import json
import numpy as np

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv
from clearml import Task
from icecream import ic

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.optim as optim
from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2Model,
                          PretrainedConfig,
                          BertConfig,
                          BertModel,
                          pipeline)

from dialRL.models import *
from dialRL.rl_train.reward_functions import *
from dialRL.rl_train.environments import DarEnv, DarPixelEnv, DarSeqEnv
from dialRL.utils import get_device
from dialRL.rl_train.callback import MonitorCallback
from dialRL.strategies import NNStrategy

torch.autograd.set_detect_anomaly(True)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# logging.getLogger('tensorflow').setLevel(logging.FATAL)
#
# tf.enable_eager_execution()

class MemoryDataset(Dataset):
    """ Customed Dataset class for our Instances data
    """
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        """ Returns a couple (image, neares 1hot)"""
        instance = self.data[idx]
        return instance


class SupervisedTrainer():
    def __init__(self, flags, sacred=None):
        ''' Inintialisation of the trainner:
                Entends to load all the correct set up, ready to train
        '''

        # Incorporate arguments to the object parameters
        for key in flags:
            setattr(self, key, flags[key])

        # Create saving experient dir
        if False :#self.sacred :
            self.path_name = '/'.join([self.sacred.experiment_info['base_dir'], self.file_dir, str(self.sacred._id)])
        else :
            self.path_name = self.rootdir + '/data/rl_experiments/' + self.alias + time.strftime("%d-%H-%M")
            print(' ** Saving train path: ', self.path_name)
            if not os.path.exists(self.path_name):
                os.makedirs(self.path_name, exist_ok=True)
            else :
                print(' Already such a path.. adding random seed')
                self.path_name = self.path_name + '#' + str(np.random.randint(1000))
                os.makedirs(self.path_name, exist_ok=True)

        # Save parameters
        with open(self.path_name + '/parameters.json', 'w') as f:
            json.dump(vars(self), f)

        self.sacred = sacred

        self.device = get_device()

        #### RL elements

        reward_function = globals()[self.reward_function]()

        ## TODO: Add globals()[self.env]
        self.env = DarSeqEnv(size=self.image_size,
                          target_population=self.nb_target,
                          driver_population=self.nb_drivers,
                          reward_function=reward_function,
                          rep_type='trans',
                          max_step=self.max_step,
                          test_env=False,
                          dataset=self.dataset,
                          verbose=self.verbose)

        # self.eval_env = DummyVecEnv([lambda : DarSeqEnv(size=self.image_size,
        self.eval_env = DarSeqEnv(size=self.image_size,
                                  target_population=self.nb_target,
                                  driver_population=self.nb_drivers,
                                  reward_function=reward_function,
                                  rep_type='trans',
                                  max_step=self.max_step,
                                  test_env=True,
                                  dataset=self.dataset)
                          # dataset=self.dataset) for i in range(1)])

        self.supervision = NNStrategy(reward_function=self.reward_function)
        self.supervision.env = self.env

        # Model Choice
        if self.model=='MlpPolicy':
            self.model = MlpPolicy
        elif self.model=='MlpLstmPolicy':
            self.model = MlpLstmPolicy
        elif self.model=='Trans1':
            self.model = globals()[self.model](src_vocab_size=50000,
                                                 trg_vocab_size=self.nb_target + 1,
                                                 max_length=self.nb_drivers+1,
                                                 src_pad_idx=self.image_size,
                                                 trg_pad_idx=self.image_size,
                                                 dropout=self.dropout,
                                                 device=self.device).to(self.device).double()
        else :
            raise "self.model in PPOTrainer is not found"

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
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.95)
        else :
            raise "Not found optimizer"

        # Scheduler
        if self.scheduler == 'plateau' :
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.patience, factor=self.gamma)
        elif self.scheduler == 'step':
            self.scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=self.gamma)

        # number of elements passed throgh the model for each epoch
        self.testing_size = self.batch_size * (10000 // self.batch_size)    #About 10k
        self.training_size = self.batch_size * (100000 // self.batch_size)   #About 100k

        self.monitor = MonitorCallback(eval_env=self.eval_env,
                                      check_freq=self.monitor_freq,
                                      save_example_freq=self.example_freq,
                                      log_dir=self.path_name,
                                      n_eval_episodes=self.eval_episodes,
                                      verbose=self.verbose,
                                      sacred=self.sacred)


        print(' *// What is this train about //* ')
        for item in vars(self):
            print(item, ':', vars(self)[item])


    def forward_data(self, data):
        print(data)
        inputs, neighbors = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
        labels = neighbors[:,0]
        shuffled_indexes = torch.randperm(neighbors.shape[1])
        anonym_neighbors = neighbors[:,shuffled_indexes].to(self.device)

        outputs = self.model(inputs.to(self.device).type(torch.LongTensor),
                        torch.tensor([[self.image_size**2] for _ in range(inputs.shape[0])]).to(self.device).type(torch.LongTensor))
        labels = label2heatmap(labels, self.image_size).to(self.device)
        labels = torch.argmax(labels, 1)
        outputs = torch.squeeze(outputs[:, :, :-1])

        return outputs, labels


    def train(self, memory):
        max_test_accuracy = 0
        running_loss = 0
        total = 0
        correct = nearest_accuracy = pointing_accuracy = 0

        self.model.train()
        for i, data in enumerate(memory):
            # set the parameter gradients to zero
            self.optimizer.zero_grad()

            observation, model_action, supervised_action = data
            loss = self.criterion(model_action, supervised_action)

            loss.backward()

            # update the gradients
            self.optimizer.step()

            total += supervised_action.size(0)
            correct += np.sum((model_action.argmax(-1) == supervised_action).cpu().numpy())
            running_loss += loss.item()

        print('-> Réussite: ', 100 * correct/total, '%')
        print('-> Loss:', running_loss)
        self.scheduler.step(running_loss)

        # if self.statistics['test_accuracy'][-1] == np.max(self.statistics['test_accuracy']) :
        #     if self.checkpoint_type == 'best':
        #         self.save_model(epoch=epoch)
        # elif self.checkpoint_type == 'all':
        #     self.save_model(epoch=epoch)



    def run(self):
        print('\t ** Learning START ! **')
        done = True

        # Run X number of training epochs
        for epoch in range(self.epochs):
            last_time = 0
            # self.env.render()
            memory = []

            # Generate a Memory batch
            for b in range(self.batch_size * 1):

                if done :
                    observation = self.env.reset()

                info_block, positions = observation
                # ic(info_block)

                model_action = self.model(torch.tensor([info_block]).type(torch.LongTensor).to(self.device),
                                          torch.tensor([[0]]).type(torch.LongTensor).to(self.device),
                                          positions=positions)
                supervised_action = self.supervision.action_choice(observation)
                supervised_action = torch.tensor([supervised_action]).type(torch.LongTensor).to(self.device)

                # print('Action', model_action.argmax(-1))
                chosen_action = model_action.argmax(-1)

                memory.append([observation, model_action[0, 0], supervised_action.squeeze(0)])

                # Arbitrary choice of following the model action to evolve the environment
                observation, reward, done, info = self.env.step(chosen_action)
                # self.env.render()

            print('Train now, with memorry size:', len(memory))
            train_data = MemoryDataset(memory)
            data = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle)
            self.train(data)

        self.env.close()

        print('\t ** Learning DONE ! **')
