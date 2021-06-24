from dialRL.strategies import CompleteRoute
from dialRL.dataset import DataFileGenerator
from dialRL.environments import DarSeqEnv
from dialRL.strategies.external.darp_rf.run_rf_algo import run_rf_algo
from dialRL.utils import get_device, objdict, SupervisionDataset
from dialRL.utils.reward_functions import  *


import os
import torch
import sys

'''
- Generate random data with an env
      - Env(param) + GenerateRandom
     - Save it as a txt  file for the desired amount.
      - Just use the datageneration script for that
     - Run the algo on that désired data, looping on the file names
      - Either bash type to start the script, either python be calling th right function (should be doable)
      - capture all the output names
     - Use all those output files to loop on a  complete route strategie and saving all the env element as data from there.
'''

class RFGenerator():

    def __init__(self, params):

        self.data_size = params.data_size
        self.timeless = params.timeless
        self.supervision_function = params.supervision_function
        self.image_size = params.image_size
        self.nb_target= params.nb_target
        self.nb_drivers = params.nb_drivers
        self.device = params.device
        self.rep_type = params.rep_type #'trans29'
        self.dataset_name = params.dataset
        self.rootdir = params.rootdir
        self.verbose = params.verbose
        self.reward_function =  globals()[params.reward_function]()
        self.last_save_size = 0

        self.gen_env = DarSeqEnv(size=self.image_size, target_population=self.nb_target, driver_population=self.nb_drivers,
                        rep_type=self.rep_type, reward_function=self.reward_function)
        self.instances_number = int(self.data_size / (self.nb_target * self.nb_drivers))
        self.dir_path = self.rootdir + '/dialRL/strategies/data/DARP_cordeau/'

        if self.dataset_name:
            data_type = dataset_name.split('/')[-1].split('.')[0]
            self.saving_name = self.rootdir + '/data/supervision_data/' + data_type + '_s{s}_tless{tt}_fun{sf}.pt'.format(s=str(self.data_size),
                                                                                                                tt=str(self.timeless),
                                                                                                                sf=str(self.supervision_function))
        else :
            self.saving_name = self.rootdir + '/data/supervision_data/' + 's{s}_t{t}_d{d}_i{i}_tless{tt}_fun{sf}.pt'.format(s=str(self.data_size),
                                                                                                              t=str(self.nb_target),
                                                                                                              d=str(self.nb_drivers),
                                                                                                              i=str(self.image_size),
                                                                                                              tt=str(self.timeless),
                                                                                                              sf=str(self.supervision_function))


    def partial_name(self, size):
        saving_name = self.rootdir + '/data/supervision_data/' + 's{s}_t{t}_d{d}_i{i}_tless{tt}_fun{sf}.pt'.format(s=str(size),
                                                                                                          t=str(self.nb_target),
                                                                                                          d=str(self.nb_drivers),
                                                                                                          i=str(self.image_size),
                                                                                                          tt=str(self.timeless),
                                                                                                          sf=str(self.supervision_function))
        return saving_name


    def generate_dataset(self):
        """
            Use an instance generator  in order to create .txt instances .
            Then used as input of the RF algorithm. We generate a .txt solution output
            This solution is finally read by a CompleteRoute strategie wrapper.
            Iterating on the instance environement, we can capture all the action at the disired time of observation.
        """
        if os.path.isfile(self.saving_name) :
            print('This data is already out there !')
            dataset = torch.load(self.saving_name)
            return dataset

        print('Going to generate a max of', self.instances_number, ' instances. Aiming to get a total of ', self.data_size, ' datapoints')

        data = []
        i = 0

        while len(data) < self.data_size and i < self.instances_number:
            # Generate 1 txt instance after an other
            file_gen = DataFileGenerator(env=self.gen_env, out_dir=self.dir_path, data_size=1)
            instance_file_name = file_gen.generate_file()[0]

            print('\t ** Solution N° ', i,' searching with RF Started for: ', instance_file_name)
            if not self.verbose:
                sys.stdout = open('test_file.out', 'w')
            try :
                solution_file = run_rf_algo('0')
            except:
                print('ERROR in RUN RF ALGO. PASSING THROUGH')
            if not self.verbose :
                sys.stdout = sys.__stdout__

            supervision_strategie = CompleteRoute(solution_file=solution_file,
                                                  size=self.image_size,
                                                  target_population=self.nb_target,
                                                  driver_population=self.nb_drivers,
                                                  reward_function='ConstantReward',
                                                  time_end=1400,
                                                  max_step=5000,
                                                  dataset=instance_file_name,
                                                  test_env=True,
                                                  recording=True)


            env = DarSeqEnv(size=self.image_size, target_population=self.nb_target, driver_population=self.nb_drivers,
                            rep_type=self.rep_type, reward_function=self.reward_function, test_env=True, dataset=instance_file_name)

            done = False
            sub_data = []
            observation = env.reset()
            supervision_strategie.env = env

            while not done:
                supervised_action = supervision_strategie.action_choice()
                supervised_action = torch.tensor([supervised_action]).type(torch.LongTensor).to(self.device)
                sub_data.append([observation, supervised_action])
                observation, reward, done, info = env.step(supervised_action)

            if env.is_fit_solution():
                data = data + sub_data
            else :
                print('/!\ Found a non feasable solution. It is not saved', env.targets_states())

            # If current data list is big enough, just save it.
            if len(data) - self.last_save_size > 500000:
                self.last_save_size = len(data)
                train_data = SupervisionDataset(data)
                torch.save(train_data, self.partial_name(size=len(data)))
                # autoremove the old version
                print('Saving data status')


            i += 1
            print('Generating data... [{i}/{ii}]'.format(i=len(data), ii=self.data_size))


        if len(data) < self.data_size :
            print('***************************************************************')
            print(' * incomplete generation... possibly an unfeasible situation * ')
            print('***************************************************************')

        print('Done Generating !')
        train_data = SupervisionDataset(data)
        torch.save(train_data, self.saving_name)
        print('Saving the data as: ', self.saving_name)
        return train_data

if __name__ == '__main__':
    rf_gen = RFGenerator(params=objdict({
        'data_size': 100,
        'timeless': False,
        'supervision_function': 'rf',
        'image_size': 4,
        'nb_target': 2,
        'nb_drivers': 1,
        'device': get_device(),
        'reward_function': ConstantReward(),
        'rep_type': 'trans29',
        'dataset': '',
        'rootdir': '/home/tibo/Documents/Prog/EPFL/own'
    }))

    rf_gen.generate_dataset()
