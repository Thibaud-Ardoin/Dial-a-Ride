import gym
import logging
import os
import imageio
import time
import json
import numpy as np
import math


# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
# from stable_baselines.common import make_vec_env
# from stable_baselines import PPO2
# from stable_baselines.common.callbacks import EvalCallback
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines.common.vec_env import DummyVecEnv
from clearml import Task
from icecream import ic
import drawSvg as draw

from moviepy.editor import *
from matplotlib.image import imsave

from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.optim as optim

# from transformers import (GPT2Tokenizer,
#                           GPT2Config,
#                           GPT2Model,
#                           PretrainedConfig,
#                           BertConfig,
#                           BertModel,
#                           pipeline)

from dialRL.strategies.external.darp_rf.run_rf_algo import run_rf_algo
from dialRL.models import *
from dialRL.utils.reward_functions import *
from dialRL.environments import DarEnv, DarPixelEnv, DarSeqEnv
from dialRL.utils import get_device, trans25_coord2int, objdict
from dialRL.dataset import DataFileGenerator
# from dialRL.rl_train.callback import MonitorCallback
# from dialRL.strategies import NNStrategy, NNStrategyV2
from dialRL.dataset import RFGenerator

torch.autograd.set_detect_anomaly(True)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# logging.getLogger('tensorflow').setLevel(logging.FATAL)
#
# tf.enable_eager_execution()



class Tester():
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
        if self.typ == 15:
            self.rep_type = 'trans15'
        elif self.typ == 16:
            self.rep_type = 'trans16'
        else :
            self.rep_type = 'trans29'

        ## TODO: Add globals()[self.gen_env]
        self.gen_env = DarSeqEnv(size=self.image_size,
                          target_population=self.nb_target,
                          driver_population=self.nb_drivers,
                          reward_function=self.reward_function,
                          rep_type=self.rep_type,
                          max_step=self.max_step,
                          test_env=False,
                          timeless=self.timeless,
                          dataset=self.dataset,
                          verbose=self.verbose)

        # self.eval_env = DummyVecEnv([lambda : DarSeqEnv(size=self.image_size,
        self.eval_env = DarSeqEnv(size=self.image_size,
                                  target_population=self.nb_target,
                                  driver_population=self.nb_drivers,
                                  reward_function=self.reward_function,
                                  rep_type=self.rep_type,
                                  max_step=self.max_step,
                                  timeless=self.timeless,
                                  test_env=True,
                                  dataset=self.dataset)
                          # dataset=self.dataset) for i in range(1)])

        self.best_eval_metric = [0, 1000] # accuracy + loss
        self.nb_target = self.gen_env.target_population
        self.nb_drivers = self.gen_env.driver_population
        self.image_size = self.gen_env.size

        if self.supervision_function == 'nn':
            self.supervision = NNStrategy(reward_function=self.reward_function,
                                      env=self.gen_env)
        elif self.supervision_function == 'nnV2':
            self.supervision = NNStrategyV2(reward_function=self.reward_function,
                                      env=self.gen_env)
        elif self.supervision_function == 'rf':
            self.supervision = RFGenerator(params=objdict(vars(self)))
        else :
            raise ValueError('Could not find the supervision function demanded: '+ self.supervision_function)

        # Model Choice
        if self.model=='MlpPolicy':
            self.model = MlpPolicy
        elif self.model=='MlpLstmPolicy':
            self.model = MlpLstmPolicy
        elif self.model=='Trans1':
            self.model = globals()[self.model](src_vocab_size=1000,
                                                 trg_vocab_size=self.nb_target + 1,
                                                 max_length=self.nb_drivers+1,
                                                 src_pad_idx=self.image_size,
                                                 trg_pad_idx=self.image_size,
                                                 dropout=self.dropout,
                                                 device=self.device).to(self.device).double()
        elif self.model=='Trans2':
            self.model = globals()[self.model](src_vocab_size=1000,
                                                 trg_vocab_size=self.nb_target + 1,
                                                 max_length=self.nb_drivers+1,
                                                 src_pad_idx=self.image_size,
                                                 trg_pad_idx=self.image_size,
                                                 dropout=self.dropout,
                                                 extremas=self.gen_env.extremas,
                                                 device=self.device).to(self.device).double()
        elif self.model=='Trans25':
            self.model = globals()[self.model](src_vocab_size=1000,
                                                 trg_vocab_size=1000,
                                                 max_length=10,
                                                 src_pad_idx=-1,
                                                 trg_pad_idx=-1,
                                                 dropout=self.dropout,
                                                 extremas=self.gen_env.extremas,
                                                 device=self.device).to(self.device).double()
        elif self.model=='Trans27':
            self.model = globals()[self.model](src_vocab_size=50000,
                                                 trg_vocab_size=self.nb_target + 1,
                                                 max_length=10,
                                                 src_pad_idx=-1,
                                                 trg_pad_idx=-1,
                                                 dropout=self.dropout,
                                                 extremas=self.gen_env.extremas,
                                                 device=self.device,
                                                 typ=self.typ).to(self.device).double()
        elif self.model=='Trans28':
            self.model = globals()[self.model](src_vocab_size=50000,
                                                 trg_vocab_size=self.vocab_size + 1,
                                                 max_length=10,
                                                 src_pad_idx=-1,
                                                 trg_pad_idx=-1,
                                                 embed_size=self.embed_size,
                                                 dropout=self.dropout,
                                                 extremas=self.gen_env.extremas,
                                                 device=self.device,
                                                 typ=self.typ,
                                                 max_time=int(self.gen_env.time_end)).to(self.device).double()
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

        # Checkpoint
        if self.checkpoint_dir :
            print(' -- -- -- -- -- Loading  -- -- -- -- -- --')
            self.model.load_state_dict(torch.load(self.rootdir + '/data/rl_experiments/' + self.checkpoint_dir).state_dict())
            print(' -- The model weights has been loaded ! --')
            print(' -----------------------------------------')

        # number of elements passed throgh the model for each epoch
        self.testing_size = self.batch_size * (10000 // self.batch_size)    #About 10k
        self.training_size = self.batch_size * (100000 // self.batch_size)   #About 100k

        # self.monitor = MonitorCallback(eval_env=self.eval_env,
        #                               check_freq=self.monitor_freq,
        #                               save_example_freq=self.example_freq,
        #                               log_dir=self.path_name,
        #                               n_eval_episodes=self.eval_episodes,
        #                               verbose=self.verbose,
        #                               sacred=self.sacred)
        self.current_epoch = 0

        #RF generation info
        self.dir_path = self.rootdir + '/dialRL/strategies/data/DARP_cordeau/'
        self.tmp_name = self.alias + time.strftime("%d-%H-%M")

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
                # if self.gen_env.__class__.__name__ == 'DummyVecEnv':
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

    def save_svg_example(self, observations, rewards, number, time_step):
        dir = self.path_name + '/example/' + str(time_step) + '/ex_number' + str(number)
        video_name = dir + '/Strat_res.mp4'
        if dir is not None:
            os.makedirs(dir, exist_ok=True)

        with draw.animate_video(video_name, align_right=True, align_bottom=True) as anim:
            # Add each frame to the animation
            for i, s in enumerate(observations):
                anim.draw_frame(s)
                if i==len(observations)-1 or i==0:
                    for i in range(5):
                        anim.draw_frame(s)

        if self.sacred :
            self.sacred.get_logger().report_media('Gif', 'Res_' + str(number) + '_Rwd=' + str(np.sum(rewards)),
                                                  iteration=time_step,
                                                  local_path=video_name)


    def run(self):
        if self.supervision_function=='rf':
            supervision = False
            self.online_evaluation(full_test=True, supervision=supervision, saving=False, rf=True)
        else :
            supervision = True
            self.online_evaluation(full_test=True, supervision=supervision)



    def online_evaluation(self, full_test=True, supervision=False, saving=True, rf=False):
        """
            Online evaluation of the model according to the supervision method.
            As it is online, we  can maximise the testing informtion about the model.
        """
        print(' ** Eval started ** ')
        correct = total = running_loss = total_reward = 0
        delivered = 0
        gap = 0
        fit_sol = 0
        i=0
        self.supervision.env = self.eval_env
        if full_test :
            eval_name = 'Test stats'
        else :
            eval_name = 'Supervised Test stats'

        self.model.eval()
        for eval_step in range(self.eval_episodes):
            done = False

            if rf :
                # Generate solution and evironement instance.
                file_gen = DataFileGenerator(env=self.gen_env, out_dir=self.dir_path, data_size=1)
                instance_file_name = file_gen.generate_file(tmp_name=self.tmp_name)[0]
                solution_file = ''

                print('\t ** Solution N° ', i,' searching with RF Started for: ', instance_file_name)
                solution_file = ''
                while not solution_file :
                    if not self.verbose:
                        sys.stdout = open('test_file.out', 'w')
                    try :
                        rf_time = time.time()
                        solution_file, supervision_perf, l_bound = run_rf_algo('0')
                        rf_time = time.time() - rf_time
                        if not self.verbose :
                            sys.stdout = sys.__stdout__
                        if l_bound is None :
                            print('Wrong solution ?')
                            solution_file = ''
                    except:
                        if not self.verbose :
                            sys.stdout = sys.__stdout__
                        print('ERROR in RUN RF ALGO. PASSING THROUGH')

                reward_function = globals()[self.reward_function]()

                self.eval_env = DarSeqEnv(size=self.image_size, target_population=self.nb_target, driver_population=self.nb_drivers,
                                          rep_type=self.rep_type, reward_function=reward_function, test_env=True, dataset=instance_file_name)
                self.eval_env.best_cost = l_bound

            observation = self.eval_env.reset()

            if saving:
                if self.example_format == 'svg':
                    to_save = [self.eval_env.get_svg_representation() if full_test else 0]
                else :
                    to_save = [self.eval_env.get_image_representation() if full_test else 0]
            save_rewards = [0]
            last_time = 0

            round_counter = 0
            trans_time = time.time()

            while not done:
                world, targets, drivers, positions, time_contraints = observation
                w_t = [torch.tensor([winfo],  dtype=torch.float64) for winfo in world]
                t_t = [[torch.tensor([tinfo], dtype=torch.float64 ) for tinfo in target] for target in targets]
                d_t = [[torch.tensor([dinfo],  dtype=torch.float64) for dinfo in driver] for driver in drivers]
                info_block = [w_t, t_t, d_t]

                positions = [torch.tensor([positions[0]], dtype=torch.float64),
                             [torch.tensor([position], dtype=torch.float64) for position in positions[1]],
                             [torch.tensor([position], dtype=torch.float64) for position in positions[2]]]

                time_contraints = [torch.tensor([time_contraints[0]], dtype=torch.float64),
                             [torch.tensor([time], dtype=torch.float64) for time in time_contraints[1]],
                             [torch.tensor([time], dtype=torch.float64) for time in time_contraints[2]]]

                target_tensor = torch.tensor([world[1]]).unsqueeze(-1).type(torch.LongTensor).to(self.device)

                model_action = self.model(info_block,
                                          target_tensor,
                                          positions=positions,
                                          times=time_contraints)

                if supervision :
                    supervised_action = self.supervision.action_choice()
                    supervised_action = torch.tensor([supervised_action]).type(torch.LongTensor).to(self.device)

                chosen_action = model_action[:, 0].argmax(-1).cpu().item()

                if full_test :
                    observation, reward, done, info = self.eval_env.step(chosen_action)
                else :
                    observation, reward, done, info = self.eval_env.step(supervised_action)

                # self.eval_env.render()
                if supervision :
                    loss = self.criterion(model_action[:,0], supervised_action)
                    running_loss += loss.item()
                else :
                    loss = 0
                    running_loss += 0

                total_reward += reward
                total += 1
                round_counter += 1
                if supervision:
                    correct += (chosen_action == supervised_action).cpu().numpy()
                else :
                    correct += 0

                if saving:
                    if self.eval_env.time_step > last_time and full_test:
                        last_time = self.eval_env.time_step
                        if self.example_format == 'svg':
                            to_save.append(self.eval_env.get_svg_representation())
                        else :
                            to_save.append(self.eval_env.get_image_representation())
                        save_rewards.append(reward)

            trans_time = time.time() - trans_time

            fit_sol += info['fit_solution'] #self.eval_env.is_fit_solution()
            delivered += info['delivered']
            if fit_sol:
                gap += info['GAP']
            print('- Supvi Total distance:', supervision_perf)
            print('- Model Total distance:', self.eval_env.total_distance)

            print('- - Supvi Time:', rf_time)
            print('- - Model Time:', trans_time)
            print('- - Passe Time:', trans_time/round_counter)

            # Saving an example
            if saving :
                if self.example_format == 'svg':
                    self.save_svg_example(to_save, save_rewards, i, time_step=self.current_epoch)
                else :
                    self.save_example(to_save, save_rewards, 0, time_step=self.current_epoch)
            i += 1


        # To spare time, only the last example is saved
        eval_acc = 100 * correct/total
        eval_loss = running_loss/total

        print('\t-->' + eval_name + 'Réussite: ', eval_acc, '%')
        print('\t-->' + eval_name + 'Loss:', running_loss/total)
        print('\t-->' + eval_name + 'Fir sol%:', 100*fit_sol/self.eval_episodes)
        print('\t-->' + eval_name + 'Average delivered:', delivered/self.eval_episodes)
        if fit_sol :
            print('\t-->' + eval_name + 'Average gap:', gap/fit_sol)
        else :
            print('\t--> No fit sol :/')
        print('\t-->' + eval_name + 'Step Reward:', total_reward/total)

        # Model saving. Condition: Better accuracy and better loss
        if eval_acc >= self.best_eval_metric[0] and eval_loss <= self.best_eval_metric[1] and full_test:
            self.best_eval_metric[0] = eval_acc
            self.best_eval_metric[1] = eval_loss
            if self.checkpoint_type == 'best':
                model_name = self.path_name + '/models/best_model.pt'
            else :
                model_name = self.path_name + '/models/model_' + str(self.current_epoch) + '.pt'
            os.makedirs(self.path_name + '/models/', exist_ok=True)
            print('\t New Best Accuracy Model <3')
            print('\tSaving as:', model_name)
            torch.save(self.model, model_name)

        # Statistics on clearml saving
        if self.sacred :
            self.sacred.get_logger().report_scalar(title=eval_name,
                series='reussite %', value=eval_acc, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title=eval_name,
                series='Loss', value=running_loss/total, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title=eval_name,
                series='Fit solution %', value=100*fit_sol/self.eval_episodes, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title=eval_name,
                series='Average delivered', value=delivered/self.eval_episodes, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title=eval_name,
                series='Average gap', value=gap/self.eval_episodes, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title=eval_name,
                series='Step Reward', value=total_reward/total, iteration=self.current_epoch)
