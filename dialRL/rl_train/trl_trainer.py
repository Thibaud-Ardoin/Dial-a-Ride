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
# import tensorflow as tf
from clearml import Task

import torch
from transformers import (GPT2Tokenizer,
                          GPT2Config,
                          GPT2Model,
                          PretrainedConfig,
                          BertConfig,
                          BertModel,
                          pipeline)

from dialRL.models import *
from dialRL.rl_train.trl.ppo_trl import PPOTrainer
from dialRL.rl_train.trl.GPT2_trl import GPT2HeadWithValueModel, respond_to_batch
from dialRL.environments import DarEnv, DarPixelEnv, DarSeqEnv
from dialRL.utils import get_device, obs2int, coord2int
from dialRL.rl_train.callback import MonitorCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# tf.enable_eager_execution()


class TrlTrainer():
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

        # RL elements

        ## TODO: Add globals()[self.env]
        self.env = DarSeqEnv(size=self.image_size,
                          target_population=self.nb_target,
                          driver_population=self.nb_drivers,
                          max_step=self.max_step,
                          test_env=False,
                          dataset=self.dataset,
                          verbose=self.verbose)

        self.eval_env = DummyVecEnv([lambda : DarSeqEnv(size=self.image_size,
                          target_population=self.nb_target,
                          driver_population=self.nb_drivers,
                          max_step=self.max_step,
                          test_env=True,
                          dataset=self.dataset) for i in range(1)])

        if self.model=='MlpPolicy':
            self.model = MlpPolicy
        elif self.model=='MlpLstmPolicy':
            self.model = MlpLstmPolicy
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
                                  save_example_freq=self.example_freq,
                                  log_dir=self.path_name,
                                  n_eval_episodes=self.eval_episodes,
                                  verbose=self.verbose,
                                  sacred=self.sacred)


        # get models

        # configuration = GPT2Config()
        # print(configuration)
        # self.gpt2_model = GPT2Model(configuration)
        # self.gpt2_model_ref = GPT2Model(configuration)

        # feature_extrator = pipeline('feature-extraction')
        # print(feature_extrator)
        # out = feature_extrator('test', size=1)
        # print(out)

        # configuration = BertConfig()
        # print(configuration)
        # self.gpt2_model = BertModel(configuration)
        # self.gpt2_model_ref = BertModel(configuration)

        # self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # self.word_embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=16)
        # self.position_embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=16)

        self.gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
        self.gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
        # self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # initialize trainer
        ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
    #         default_params = {
    #     "lr": 1.41e-5,
    #     "adap_kl_ctrl": True,
    #     "init_kl_coef":0.2,
    #     "target": 6,
    #     "horizon":10000,
    #     "gamma":1,
    #     "lam":0.95,
    #     "cliprange": .2,
    #     "cliprange_value":.2,
    #     "vf_coef":.1,
    #     "batch_size": 256,
    #     "forward_batch_size": 16,
    #     "ppo_epochs": 4,
    # }
        self.ppo_trainer = PPOTrainer(self.gpt2_model, self.gpt2_model_ref, **ppo_config)

        print(' *// What is this train about //* ')
        for item in vars(self):
            print(item, ':', vars(self)[item])


    def run(self):
        print('\t ** Learning START ! **')

        training_steps = 0

        # For i in range traning step
        for i in range(self.epochs):
            # Do we need it ?
            memorry = []
            obs = self.env.reset()
            done = False
            while not done :
                obs = obs2int(obs)

                #Traduction to query
                query_tensors = torch.tensor([obs])
                response_tensor  = respond_to_batch(self.gpt2_model, query_tensors, txt_len=1)
                # print(response_tensor)

                # traduction to action
                action = response_tensor[0][0]
                obs, rwd, done, info = self.env.step(action)

                # Traduction to reward
                reward = torch.tensor([float(rwd)])

                # train model with ppo / batch segmentation ???
                train_stats = self.ppo_trainer.step(query_tensors, response_tensor, reward)
                for key in train_stats:
                    self.sacred.get_logger().report_scalar(title='Train stats',
                        series=key, value=train_stats[key], iteration=training_steps)

                training_steps += 1
                print(train_stats)

        print('\t ** Learning DONE ! **')


    def test(self):
        pass
        # model_name = self.checkpoint_dir
        # if model_name == '':
        #     model_name = self.path_name + '/best_model.zip'
        #
        # model = PPO2.load(model_name)
        #
        # evaluate_policy(model=model,
        #                 env=self.eval_env,
        #                 n_eval_episodes=self.eval_episodes,
        #                 render=True)
        #
        # #TODO: Write a testing belt for a RL ppo environment


if __name__=='__main__':
    trainer = TrlTrainer()
    trainer.run()
