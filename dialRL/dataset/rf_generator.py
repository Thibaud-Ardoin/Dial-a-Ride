from dialRL.strategies import CompleteRoute
from dialRL.dataset import DataFileGenerator
from dialRL.environments import DarSeqEnv
from dialRL.strategies.external.darp_rf.run_rf_algo import run_rf_algo
from dialRL.utils import get_device, objdict, SupervisionDataset
from dialRL.utils.reward_functions import  *

from torch.utils.data import ConcatDataset, ChainDataset
import os
import torch
import sys
import numpy as np

from icecream import ic

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
        self.datadir = params.datadir
        self.typ = params.typ
        self.reward_function =  globals()[params.reward_function]()
        self.last_save_size = 0
        self.data_part = 1

        self.gen_env = DarSeqEnv(size=self.image_size, target_population=self.nb_target, driver_population=self.nb_drivers,
                        rep_type=self.rep_type, reward_function=self.reward_function)
        self.instances_number = int(self.data_size / (self.nb_target * self.nb_drivers))
        self.dir_path = self.rootdir + '/dialRL/strategies/data/DARP_cordeau/'

        if self.datadir:
            data_directory = self.datadir
        else :
            data_directory = self.rootdir + '/data/supervision_data/'

        if self.dataset_name:
            data_type = dataset_name.split('/')[-1].split('.')[0]
            self.saving_name = data_directory + data_type + '_s{s}_tless{tt}_fun{sf}_typ{ty}/'.format(s=str(self.data_size),
                                                                                                                tt=str(self.timeless),
                                                                                                                sf=str(self.supervision_function),
                                                                                                                ty=str(self.rep_type))
        else :
            self.saving_name = data_directory + 's{s}_t{t}_d{d}_i{i}_tless{tt}_fun{sf}_typ{ty}/'.format(s=str(self.data_size),
                                                                                                              t=str(self.nb_target),
                                                                                                              d=str(self.nb_drivers),
                                                                                                              i=str(self.image_size),
                                                                                                              tt=str(self.timeless),
                                                                                                              sf=str(self.supervision_function),
                                                                                                              ty=str(self.rep_type))
        self.tmp_name = self.saving_name.split('/')[-2]


    def partial_name(self, size):
        saving_name = self.saving_name + '/dataset_elementN' + str(self.data_part) + '_size' + str(size) + '.pt'
        self.data_part += 1
        return saving_name


    def load_dataset(self):
        files_names = os.listdir(self.saving_name)
        return [self.saving_name + file for file in os.listdir(self.saving_name)]

        datasets = []
        for file in files_names:
            print('Datafile folder:', self.saving_name)
            print(file)
            datasets.append(torch.load(self.saving_name + file))

        # data, sup = datasets[0][0]
        # world, targets, drivers, positions, time_constraints = data
        # ic(world, targets, drivers, positions, time_constraints)
        # data, sup = datasets[1][0]
        # world, targets, drivers, positions, time_constraints = data
        # ic(world, targets, drivers, positions, time_constraints)

        datasets = self.refill_data(datasets)
        # for i in range(len(datasets[0][0][0])):
        #     ic(datasets[0][0][0][i])
        #     ic(datasets[1][0][0][i])
        # for i in range(len(datasets[0][0][0])):
        #     ic(np.shape(datasets[0][0][0][i]))
        #     ic(np.shape(datasets[1][0][0][i]))
        return ConcatDataset(datasets)

    def refill_data(self, datasets):
        new_datasets = []
        for sub_dataset in datasets:
            current_target_size = len(sub_dataset[0][0][1])
            current_driver_size = len(sub_dataset[0][0][2])

            target_to_add = self.nb_target - current_target_size
            driver_to_add = self.nb_drivers - current_driver_size
            if target_to_add > 0 or driver_to_add > 0:
                new_data = []
                # Add extra zeros to match desired size.
                for data in sub_dataset:
                    world, targets, drivers, positions, time_constraints = data[0]

                    # Adding Zeros to targets
                    targets = targets + list(map(list, list(np.zeros((target_to_add, 3), dtype=np.float64))))
                    positions[1] = positions[1] + list(np.zeros((target_to_add, 4), dtype=np.float64))
                    time_constraints[1] = time_constraints[1] + list(np.zeros((target_to_add, 4), dtype=np.float64))
                    # Adding Zeros to drivers
                    drivers = drivers + np.zeros((driver_to_add, len(drivers[-1])), dtype=np.float64).tolist()
                    positions[2] = positions[2] + list(np.zeros((driver_to_add, 2), dtype=np.float64))
                    time_constraints[2].append(np.float64(0))

                    obs = world, targets, drivers, positions, time_constraints
                    new_data.append([obs, data[1]])

                new_datasets.append(new_data)

            elif target_to_add < 0 :
                # Undesired case
                raise "Error of desired size < target size in the data"

            elif target_to_add == 0:
                # Perfect data. Don't tuch it
                new_datasets.append(sub_dataset)

        return new_datasets


    def generate_dataset(self):
        """
            Use an instance generator  in order to create .txt instances .
            Then used as input of the RF algorithm. We generate a .txt solution output
            This solution is finally read by a CompleteRoute strategie wrapper.
            Iterating on the instance environement, we can capture all the action at the disired time of observation.
        """
        if os.path.isdir(self.saving_name) :
            print('This data is already out there ! (at least a part of it) Loading it ...')
            return self.load_dataset()
        else :
            os.makedirs(self.saving_name)

        print('Going to generate a max of', self.instances_number, ' instances. Aiming to get a total of ', self.data_size, ' datapoints')

        data = []
        i = 0

        while (self.last_save_size + len(data) < self.data_size) and i < self.instances_number: #len(data) < self.data_size and i < self.instances_number:
            # Generate 1 txt instance after an other
            file_gen = DataFileGenerator(env=self.gen_env, out_dir=self.dir_path, data_size=1)
            instance_file_name = file_gen.generate_file(tmp_name=self.tmp_name)[0]
            solution_file = ''

            print('\t ** Solution N° ', i,' searching with RF Started for: ', instance_file_name)
            if not self.verbose:
                sys.stdout = open('test_file.out', 'w')
            try :
                solution_file, perf, l_bound = run_rf_algo('0')
            except:
                print('ERROR in RUN RF ALGO. PASSING THROUGH')
            if not self.verbose :
                sys.stdout = sys.__stdout__

            if solution_file :
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

                # If current data list is big enough, save it as a dataset_element.
                if sys.getsizeof(data) > 100000: #200k bytes.
                    self.last_save_size += len(data)
                    train_data = SupervisionDataset(data)
                    saving_name = self.partial_name(len(data))
                    torch.save(train_data, saving_name)
                    data = []
                    print('Saving data status')

                i += 1
                print('Generating data... [{i}/{ii}] memorry:{m}'.format(i=self.last_save_size + len(data), ii=self.data_size, m=sys.getsizeof(data)))

        if len(data) > 0 :
            train_data = SupervisionDataset(data)
            saving_name = self.partial_name(len(data))
            torch.save(train_data, saving_name)
            print('Last data element in ', self.saving_name)

        if len(data) + self.last_save_size < self.data_size :
            print('***************************************************************')
            print(' * incomplete generation... possibly an unfeasible situation * ')
            print('***************************************************************')

        print('Done Generating !')
        return self.load_dataset()


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
