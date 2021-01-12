import sys
from sacred import cli_option, Experiment
from sacred.observers import FileStorageObserver, TinyDbObserver, MongoObserver
import argparse

from utils import objdict
from train import Trainer
from instances import PixelInstance


ex = Experiment("svm")

ex.observers.append(FileStorageObserver("my_runs"))
# ex.observers.append(MongoObserver(db_name='experiments_db'))

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

    print(parser.parse_known_args(args)[0])

    return parser.parse_known_args(args)[0]

@ex.config  # Configuration is defined through local variables.
def cfg(parameters):
    parameters = parameters
    # epochs= 2000
    # model = 'UpAE'
    # scheduler = 'plateau'
    # data = '/home/ardoin/Dial-a-Ride/data/instances/split3_1nn_500k_n2_s50_m0'
    # patience = 10
    # lr = 0.001
    # criterion = 'crossentropy'
    # input_type = 'map'
    # output_type = 'map'
    # alias = 'AE'
    # batch_size = 128
    # shuffle = True
    # optimizer = 'Adam'
    # checkpoint_type = 'best'
    # milestones = [50]
    # gamma = 0.1
    # checkpoint_dir = ''
    # layers = 256
    # channels = 1

@ex.capture
def get_trainer(parameters):
    return Trainer(parameters)


@ex.automain  # Using automain to enable command line integration.
def run():
    # Get params
    param = get_args(sys.argv[1:])
    parameters = cfg(param)['parameters']
    print(parameters)

    # Get the trainer object
    trainer = get_trainer(parameters)

    # Start a train
    trainer.run()

    # Start evaluation
    trainer.evaluation()
