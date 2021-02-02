import sys
import argparse

from clearml import Task

from utils import objdict
from train import Trainer
from instances import PixelInstance
import models



def get_args(args):
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a train.",
        epilog="python train.py --epochs INT")
    # required input parameters
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--alias', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--data', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--shuffle', type=bool)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--criterion', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--checkpoint_type', type=str)
    parser.add_argument('--milestones', nargs='+', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--input_type', type=str, default='map')
    parser.add_argument('--output_type', type=str, default='coord')
    parser.add_argument('--layers', type=int)
    parser.add_argument('--channels', type=int)

    return parser.parse_known_args(args)[0]


def configg():
    param = type('', (), {})()
    param.epochs= 2000
    param.model = 'UpAE'
    param.scheduler = 'plateau'
    param.data = './data/instances/split3_1nn_1k_n2_s50_m0'
    param.patience = 10
    param.lr = 0.001
    param.criterion = 'crossentropy'
    param.input_type = 'map'
    param.output_type = 'map'
    param.alias = 'test_AE'
    param.batch_size = 128
    param.shuffle = True
    param.optimizer = 'Adam'
    param.checkpoint_type = 'best'
    param.milestones = [50]
    param.gamma = 0.1
    param.checkpoint_dir = ''
    param.layers = 256
    param.channels = 1
    param.upscale_factor = 2
    return param

# @ex.config  # Configuration is defined through local variables.
# def cfg():
#     # parameters = parameters
#     epochs= 2000
#     model = 'UpAE'
#     scheduler = 'plateau'
#     data = './data/instances/split3_1nn_1k_n2_s50_m0'
#     patience = 10
#     lr = 0.001
#     criterion = 'crossentropy'
#     input_type = 'map'
#     output_type = 'map'
#     alias = 'test_AE'
#     batch_size = 128
#     shuffle = True
#     optimizer = 'Adam'
#     checkpoint_type = 'best'
#     milestones = [50]
#     gamma = 0.1
#     checkpoint_dir = ''
#     layers = 256
#     channels = 1

def get_trainer(parameters):
    return Trainer(parameters)


def goooo():

    # Get params
    parameters = configg()#get_args(sys.argv[1:])

    # Get the trainer object
    trainer = get_trainer(parameters)

    # Start a train
    trainer.run()

    # Start evaluation
    trainer.evaluation()


if __name__ == '__main__':
    task = Task.init(
        project_name="DaRP", task_name="1st experiment")
    goooo()
