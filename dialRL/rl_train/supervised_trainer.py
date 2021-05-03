import gym
import logging
import os
import imageio
import time
import json
import numpy as np
import math

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv
from clearml import Task
from icecream import ic

from moviepy.editor import *
from matplotlib.image import imsave

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
from dialRL.utils import get_device, trans25_coord2int
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

class SupervisionDataset(Dataset):
    """ Customed Dataset class for our Instances data
    """
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ simple idx """
        return self.data[idx]


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
                          rep_type='trans25',
                          max_step=self.max_step,
                          test_env=True,
                          timeless=True,
                          dataset=self.dataset,
                          verbose=self.verbose)

        # self.eval_env = DummyVecEnv([lambda : DarSeqEnv(size=self.image_size,
        self.eval_env = DarSeqEnv(size=self.image_size,
                                  target_population=self.nb_target,
                                  driver_population=self.nb_drivers,
                                  reward_function=reward_function,
                                  rep_type='trans25',
                                  max_step=self.max_step,
                                  timeless=True,
                                  test_env=True,
                                  dataset=self.dataset)
                          # dataset=self.dataset) for i in range(1)])

        self.supervision = NNStrategy(reward_function=self.reward_function,
                                      env=self.env)

        # Model Choice
        if self.model=='MlpPolicy':
            self.model = MlpPolicy
        elif self.model=='MlpLstmPolicy':
            self.model = MlpLstmPolicy
        elif self.model=='Trans1':
            self.model = globals()[self.model](src_vocab_size=70000,
                                                 trg_vocab_size=self.nb_target + 1,
                                                 max_length=self.nb_drivers+1,
                                                 src_pad_idx=self.image_size,
                                                 trg_pad_idx=self.image_size,
                                                 dropout=self.dropout,
                                                 device=self.device).to(self.device).double()
        elif self.model=='Trans2':
            self.model = globals()[self.model](src_vocab_size=70000,
                                                 trg_vocab_size=self.nb_target + 1,
                                                 max_length=self.nb_drivers+1,
                                                 src_pad_idx=self.image_size,
                                                 trg_pad_idx=self.image_size,
                                                 dropout=self.dropout,
                                                 extremas=self.env.extremas,
                                                 device=self.device).to(self.device).double()
        elif self.model=='Trans25':
            self.model = globals()[self.model](src_vocab_size=70000,
                                                 trg_vocab_size=70000,
                                                 max_length=10,
                                                 src_pad_idx=-1,
                                                 trg_pad_idx=-1,
                                                 dropout=self.dropout,
                                                 extremas=self.env.extremas,
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
        self.current_epoch = 0


        print(' *// What is this train about //* ')
        for item in vars(self):
            print(item, ':', vars(self)[item])


    def save_example(self, observations, rewards, number, time_step):
        noms = []
        dir = self.path_name + '/example/' + str(time_step) + '/ex_number' + str(number)
        if dir is not None:
            os.makedirs(dir, exist_ok=True)

            for i, obs in enumerate(observations):
                save_name = dir + '/' + str(i) + '_r=' + str(rewards[i]) + '.png'  #[np.array(img) for i, img in enumerate(images)
                # if self.env.__class__.__name__ == 'DummyVecEnv':
                #     image = self.norm_image(obs[0], scale=1)
                # else :
                #     image = self.norm_image(obs, scale=1)
                image = obs
                # print('SHae after image', np.shape(image))
                imsave(save_name, image)
                noms.append(save_name)

        # Save the imges as video
        video_name = dir + 'r=' + str(np.sum(rewards)) + '.mp4'
        clips = [ImageClip(m).set_duration(0.2)
              for m in noms]

        concat_clip = concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(video_name, fps=24, verbose=None, logger=None)

        if self.sacred :
            self.sacred.get_logger().report_media('video', 'Res_' + str(number) + '_Rwd=' + str(np.sum(rewards)),
                                                  iteration=time_step,
                                                  local_path=video_name)
        del concat_clip
        del clips


    def bidim2int(self, coordonate):
        src_vocab_size = 70000
        siderange = int(math.sqrt(src_vocab_size))
        boxh, boxw = abs(self.env.extremas[2] - self.env.extremas[0]) / siderange, abs(self.env.extremas[3] - self.env.extremas[1]) / siderange

        h = int(abs(coordonate[0] - self.env.extremas[0]) / boxh )
        w = int(abs(coordonate[1] - self.env.extremas[1]) / boxw )
        return int(h + (w * siderange))

    def int2bidim(self, int_position):
        src_vocab_size = 70000
        siderange = int(math.sqrt(src_vocab_size))
        boxh, boxw = abs(self.env.extremas[2] - self.env.extremas[0]) / siderange, abs(self.env.extremas[3] - self.env.extremas[1]) / siderange

        h = int_position - siderange * (int_position // siderange)
        w = (int_position - h) / siderange
        h = (h * boxh) + self.env.extremas[0]
        w = (w * boxw) + self.env.extremas[1]
        return h, w


    def generate_supervision_data(self):
        print('\t ** Generation Started **')
        number_batch = self.data_size // self.batch_size
        size = number_batch * self.batch_size

        data = []
        saving_name = self.rootdir + '/data/supervision_data/' + 's{s}_t{t}_d{d}_i{i}.pt'.format(s=str(size),
                                                                                              t=str(self.nb_target),
                                                                                              d=str(self.nb_drivers),
                                                                                              i=str(self.image_size))
        done = True

        if os.path.isfile(saving_name) :
            print('This data is already out there !')
            dataset = torch.load(saving_name)
            return dataset

        # Generate a Memory batch
        for element in range(size):

            if done :
                observation = self.env.reset()

            supervised_action = self.supervision.action_choice()
            supervised_action = torch.tensor([supervised_action]).type(torch.LongTensor).to(self.device)

            action_target = self.env.targets[supervised_action - 1]
            if action_target.state < 0:
                supervised_action = self.bidim2int(action_target.pickup)
            else :
                supervised_action = self.bidim2int(action_target.dropoff)

            data.append([observation, supervised_action])
            observation, reward, done, info = self.env.step(supervised_action)

            if element % 1000 == 0:
                print('Generating data... [{i}/{ii}]'.format(i=element, ii=size))

        print('Done Generating !')
        train_data = SupervisionDataset(data)
        torch.save(train_data, saving_name)
        return train_data


    def train(self, dataloader):
        max_test_accuracy = 0
        running_loss = 0
        total = 0
        correct = nearest_accuracy = pointing_accuracy = 0

        self.model.train()
        for i, data in enumerate(dataloader):
            # set the parameter gradients to zero
            self.optimizer.zero_grad()

            observation, supervised_action = data

            world, targets, drivers, positions = observation
            info_block = [world, targets, drivers]
            # Current player as trg elmt
            target_tensor = world[1].unsqueeze(-1).type(torch.LongTensor).to(self.device)
            # target_tensor = torch.tensor([0 for _ in range(self.batch_size)]).unsqueeze(-1).type(torch.LongTensor).to(self.device)
            # coord_int = trans25_coord2int(positions[1][supervised_action])

            model_action = self.model(info_block,
                                      target_tensor,
                                      positions=positions)

            # model_action = model_action[:,0]
            supervised_action = supervised_action.to(self.device)

            # loss = self.criterion(model_action, supervised_action.squeeze(-1))
            loss = self.criterion(model_action.squeeze(1), supervised_action.squeeze(-1))

            loss.backward()

            # update the gradients
            self.optimizer.step()

            total += supervised_action.size(0)
            correct += np.sum((model_action.squeeze(1).argmax(-1) == supervised_action.squeeze(-1)).cpu().numpy())
            running_loss += loss.item()

            # Limit train passage to 20 rounds
            if i == 20:
                break

        acc = 100 * correct/total
        print('-> Réussite: ', acc, '%')
        print('-> Loss:', 100*running_loss/total)
        self.scheduler.step(running_loss)

        if self.sacred :
            self.sacred.get_logger().report_scalar(title='Train stats',
                series='train loss', value=100*running_loss/total, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title='Train stats',
                series='Train accuracy', value=acc, iteration=self.current_epoch)


    def run(self):
        dataset = self.generate_supervision_data()
        print('\t ** Learning START ! **')
        done = True

        supervision_data = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.train(supervision_data)
            self.evaluate()

        print('\t ** Learning DONE ! **')


    def evaluate(self):
        correct = total = running_loss = total_reward = 0
        fit_sol = 0
        self.supervision.env = self.eval_env

        self.model.eval()
        for eval_step in range(self.eval_episodes):
            done = False
            observation = self.eval_env.reset()

            to_save = [self.eval_env.get_image_representation()]
            save_rewards = [0]
            last_time = 0

            while not done :
                world, targets, drivers, positions = observation
                w_t = [torch.tensor([winfo],  dtype=torch.float64) for winfo in world]
                t_t = [[torch.tensor([tinfo], dtype=torch.float64 ) for tinfo in target] for target in targets]
                d_t = [[torch.tensor([dinfo],  dtype=torch.float64) for dinfo in driver] for driver in drivers]
                info_block = [w_t, t_t, d_t]

                positions = [torch.tensor([positions[0]], dtype=torch.float64),
                             [torch.tensor([position], dtype=torch.float64) for position in positions[1]],
                             [torch.tensor([position], dtype=torch.float64) for position in positions[2]]]

                target_tensor = torch.tensor([world[1]]).unsqueeze(-1).type(torch.LongTensor).to(self.device)

                model_action = self.model(info_block,
                                          target_tensor,
                                          positions=positions)

                supervised_action = self.supervision.action_choice()
                supervised_action = torch.tensor([supervised_action]).type(torch.LongTensor).to(self.device)

                chosen_position = model_action[:, 0].argmax(-1).cpu().item()
                chosen_action = self.eval_env.nearest_target(self.int2bidim(chosen_position))

                observation, reward, done, info = self.eval_env.step(chosen_action)
                # self.env.render()
                # loss = self.criterion(model_action[0], supervised_action)

                total_reward += reward
                total += 1
                correct += (chosen_action == supervised_action).cpu().numpy()
                running_loss += 0 #loss.item()

                if self.eval_env.time_step > last_time :
                    # last_time = self.eval_env.time_step
                    to_save.append(self.eval_env.get_image_representation())
                    save_rewards.append(correct)


            fit_sol += self.eval_env.is_fit_solution()
            self.save_example(to_save, save_rewards, 0, time_step=self.current_epoch)

        print('\t--> Test Réussite: ', 100 * correct[0]/total, '%')
        print('\t--> Test Loss:', running_loss/total)

        if self.sacred :
            self.sacred.get_logger().report_scalar(title='Test stats',
                series='reussite %', value=100*correct[0]/total, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title='Test stats',
                series='Loss', value=running_loss/total, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title='Test stats',
                series='Fit solution %', value=100*fit_sol/self.eval_episodes, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title='Test stats',
                series='Step Reward', value=total_reward/total, iteration=self.current_epoch)
