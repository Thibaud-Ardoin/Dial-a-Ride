import numpy as np
import os
from icecream import ic

from dialRL.rl_train.environments import DarSeqEnv
# from dialRL.utils import get_device, trans25_coord2int
# from dialRL.strategies import NNStrategy

params = {
    'size': 4,
    'target_population': 5,
    'driver_population': 2,
    'reward_function': 'ConstantReward',
    'time_end': 1400,
    'max_step': 2000,
    'timeless' : False,
    'dataset': '/home/tibo/Documents/Prog/EPFL/own/data/instances/cordeau2003/tabu1.txt',
    'test_env': False,
    'out_dir': './data/formated/1',
    'data_size': 2
}


env = DarSeqEnv(size=params['size'],
                        target_population=params['target_population'],
                        driver_population=params['driver_population'],
                        reward_function=params['reward_function'],
                        time_end=params['time_end'],
                        max_step=params['max_step'],
                        timeless=params['timeless'],
                        dataset=params['dataset'],
                        test_env=params['test_env'])



os.makedirs(params['out_dir'], exist_ok=True)

# Number of vehicles, Number of requests, Maximum route duration, Vehicle capacity, Maximum ride time
#   3 48 480 6 90
#  Depot info
#   0   -1.044    2.000  0  0    0 1440
#   <id> <x> <y> <service time> <demand> <TW start> <TW end>
#   1   -2.973    6.414 10  1    0 1440

for n in range(params['data_size']):
    observation = env.reset()
    text_lists = []

    text_lists.append([env.driver_population, 2 * env.target_population, 480, env.drivers[0].max_capacity, 90])
    text_lists.append([0, env.depot_position[0], env.depot_position[1], 0, 0, 0, int(env.time_end)])
    for target in env.targets:
        text_lists.append([target.identity, target.pickup[0], target.pickup[1], 10, 1, int(target.start_fork[0]), int(target.start_fork[1])])
    for target in env.targets:
        text_lists.append([target.identity + len(env.targets), target.dropoff[0], target.dropoff[1], 10, -1, int(target.end_fork[0]), int(target.end_fork[1])])
    env.close()


    # Join the text info into 1 file data

    text_lists2 = []
    for text in text_lists :
        text_lists2.append('\t'.join(map(str, map(lambda x: round(x, 3), text))))

    final_string = '\n'.join(text_lists2)

    print(final_string)

    with open(params['out_dir'] + '/' + str(n) + '.txt', 'w') as write_file:
        write_file.write(final_string)
    write_file.close()

with open(params['out_dir'] + '/INDEX.txt', 'w') as write_file:
    for n in range(params['data_size']):
        write_file.write(str(n) + '\n')

write_file.close()
