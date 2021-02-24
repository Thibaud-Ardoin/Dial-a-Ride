import sys
import argparse
import numpy as np

from clearml import Task

from dialRL.utils import objdict
from dialRL.rl_train import PPOTrainer



def get_args(args):
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a train.",
        epilog="python train.py --epochs INT")
    # required input parameters
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--alias', default='RL_ppo2', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--optimizer', default='RMSprop', type=str)
    parser.add_argument('--criterion', default='crossentropy', type=str)
    parser.add_argument('--model', default='MlpPolicy', type=str)
    parser.add_argument('--checkpoint_type', default='best', type=str)
    parser.add_argument('--milestones', default=[50], nargs='+', type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--scheduler', default='plateau', type=str)
    parser.add_argument('--checkpoint_dir', default='', type=str)
    parser.add_argument('--input_type', type=str, default='flatmap')
    parser.add_argument('--output_type', type=str, default='flatmap')
    parser.add_argument('--layers', default=[64, 64], nargs='+', type=int)
    parser.add_argument('--total_timesteps', default=10000, type=int)
    parser.add_argument('--monitor_freq', default=1000, type=int)
    parser.add_argument('--eval_episodes', default=4, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--nb_target', default=5, type=int)
    parser.add_argument('--image_size', default=4, type=int)
    parser.add_argument('--nb_drivers', default=1, type=int)



    return parser.parse_known_args(args)[0]

def goooo():
    # Get params
    parameters = objdict(vars(get_args(sys.argv[1:])))

    # Get the trainer object
    trainer = PPOTrainer(parameters)

    # Start a train
    trainer.run()

    # Start evaluation
    trainer.test()


if __name__ == '__main__':
    numbah = np.random.randint(10000)
    task = Task.init(
        project_name="DaRP", task_name="experiment" + str(numbah))
    goooo()
