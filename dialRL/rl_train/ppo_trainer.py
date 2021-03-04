import gym
import logging
import os
import imageio
import time
import json
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.evaluation import evaluate_policy
import tensorflow as tf
from clearml import Task

from dialRL.models import *
from dialRL.rl_train.environments import DarEnv, DarPixelEnv, DarSeqEnv
from dialRL.utils import get_device
from dialRL.rl_train.callback import MonitorCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

tf.enable_eager_execution()


class PPOTrainer():
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
            self.path_name = './data/rl_experiments/' + self.alias + time.strftime("%d-%H-%M")
            print(' ** Saving train path: ', self.path_name)
            if not os.path.exists(self.path_name):
                os.makedirs(self.path_name)
            else :
                print(' Already such a path.. adding random seed')
                self.path_name = self.path_name + '#' + str(np.random.randint(1000))
                os.makedirs(self.path_name)

        # Save parameters
        with open(self.path_name + '/parameters.json', 'w') as f:
            json.dump(vars(self), f)

        self.sacred = sacred

        self.device = get_device()

        # RL elements

        ## TODO: Add globals()[self.env]
        self.env = DarSeqEnv(size=self.image_size,
                          target_population=self.nb_target,
                          driver_population=self.nb_drivers,
                          max_step=self.max_step,
                          dataset=self.dataset)

        self.eval_env = DarSeqEnv(size=self.image_size,
                          target_population=self.nb_target,
                          driver_population=self.nb_drivers,
                          max_step=self.max_step,
                          dataset=self.dataset)

        if self.model=='MlpPolicy':
            self.model = MlpPolicy
        else :
            raise "self.model in PPOTrainer is not found"
        # elif self.model=='LnMlpPolicy':
        #     self.model = LnMlpPolicy(layers=self.layers)

        # Loading model from dir
        # if self.checkpoint_dir :
        #     self.model.load_state_dict(torch.load(self.checkpoint_dir + '/best_model.pt'), strict=False)
                #'./data/experiments/' + self.checkpoint_dir + '/best_model.pt'))

        # number of elements passed throgh the model for each epoch
        self.testing_size = self.batch_size * (10000 // self.batch_size)    #About 10k
        self.training_size = self.batch_size * (100000 // self.batch_size)   #About 100k

        self.monitor = MonitorCallback(eval_env=self.eval_env,
                                  check_freq=self.monitor_freq,
                                  log_dir=self.path_name,
                                  n_eval_episodes=self.eval_episodes,
                                  verbose=self.verbose,
                                  sacred=self.sacred)

        self.ppo_model = PPO2(self.model, self.env, verbose=0,
                              policy_kwargs={'layers': self.layers})

        print(' *// What is this train about //* ')
        for item in vars(self):
            print(item, ':', vars(self)[item])


    def run(self):
        print('\t ** Learning START ! **')
        self.ppo_model.learn(total_timesteps=self.total_timesteps, callback=self.monitor)
        print('\t ** Learning DONE ! **')


    def test(self):
        model_name = self.checkpoint_dir
        if model_name == '':
            model_name = self.path_name + '/best_model.zip'

        model = PPO2.load(model_name)

        evaluate_policy(model=model,
                        env=self.eval_env,
                        n_eval_episodes=self.eval_episodes,
                        render=True)

        #TODO: Write a testing belt for a RL ppo environment
