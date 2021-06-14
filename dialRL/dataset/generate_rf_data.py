from dialRL.strategies import CompleteRoute
from dialRL.dataset import DataFileGenerator
from dialRL.rl_train.environments import DarSeqEnv
from dialRL.strategies.external.darp_rf.run_rf_algo import run_rf_algo
from dialRL.rl_train.supervised_trainer import SupervisionDataset
from dialRL.rl_train import ConstantReward
from dialRL.utils import get_device


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


data_size = 10000
timeless = False
supervision_function = 'rf'
image_size = 4
nb_target= 24
nb_drivers = 3
device = get_device()
reward_function = ConstantReward()

instances_number = int(data_size / (nb_target * nb_drivers))
print('Going to generate ', instances_number, ' instances.')


# Generate random data in text file and save those files name
env = DarSeqEnv(size=image_size, target_population=nb_target, driver_population=nb_drivers,
                rep_type='trans29', reward_function=reward_function)
rootdir = '/home/tibo/Documents/Prog/EPFL/own/'
dir = 'dialRL/strategies/data/DARP_cordeau/'
dataset_name=''

print('\t ** Generation Started **')

file_gen = DataFileGenerator(env=env, out_dir=rootdir+dir, data_size=instances_number)
file_names = file_gen.generate_file()

data = []

for i, data_file in enumerate(file_names) :
# Run rf Algo on the txt data
    sys.stdout = open('test_file.out', 'w')
    solution_file = run_rf_algo(str(i))
    sys.stdout = sys.__stdout__

    supervision_strategie = CompleteRoute(solution_file=solution_file,
                                          size=data_size,
                                          target_population=nb_target,
                                          driver_population=nb_drivers,
                                          reward_function='HybridProportionalReward',
                                          time_end=1400,
                                          max_step=5000,
                                          dataset=data_file,
                                          test_env=True,
                                          recording=True)

#####
    if dataset_name:
        data_type = dataset_name.split('/')[-1].split('.')[0]
        saving_name = rootdir + '/data/supervision_data/' + data_type + '_s{s}_tless{tt}_fun{sf}.pt'.format(s=str(data_size),
                                                                                                            tt=str(timeless),
                                                                                                            sf=str(supervision_function))
    else :
        saving_name = rootdir + '/data/supervision_data/' + 's{s}_t{t}_d{d}_i{i}_tless{tt}_fun{sf}.pt'.format(s=str(data_size),
                                                                                                          t=str(nb_target),
                                                                                                          d=str(nb_drivers),
                                                                                                          i=str(image_size),
                                                                                                          tt=str(timeless),
                                                                                                          sf=str(supervision_function))
    if os.path.isfile(saving_name) :
        print('This data is already out there !')
        dataset = torch.load(saving_name)
        # return dataset
        exit()

    env = DarSeqEnv(size=image_size, target_population=nb_target, driver_population=nb_drivers,
                    rep_type='trans29', reward_function=reward_function, test_env=True, dataset=data_file)

    done = False
    sub_data = []
    observation = env.reset()
    supervision_strategie.env = env

    print('- Instance n°', i, 'name:', data_file)
    print(' - Solustion File:', solution_file)

    # Generate a Memory batch
    for element in range(data_size):

        if done :
            if env.is_fit_solution():
                data = data + sub_data
            else :
                print(env.targets_states())
                print('/!\ Found a non feasable solution. It is not saved')
            observation = env.reset()
            sub_data = []
            break;

        supervised_action = supervision_strategie.action_choice()
        supervised_action = torch.tensor([supervised_action]).type(torch.LongTensor).to(device)

        sub_data.append([observation, supervised_action])
        observation, reward, done, info = env.step(supervised_action)

    print('Generating data... [{i}/{ii}]'.format(i=len(data), ii=data_size))


print('Done Generating !')
train_data = SupervisionDataset(data)
torch.save(train_data, saving_name)
print(saving_name)
