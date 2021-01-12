from sacred import cli_option, Experiment
from sacred.observers import FileStorageObserver

from utils import objdict
from train import Trainer
from generator import PixelInstance


ex = Experiment("svm")

ex.observers.append(FileStorageObserver("my_runs"))


@ex.config  # Configuration is defined through local variables.
def cfg():
    epochs= 2000
    model = 'UpCNN1'
    scheduler = 'plateau'
    data = '/home/ardoin/Dial-a-Ride/data/instances/split3_1nn_500k_n2_s50'
    patience = 50
    lr = 0.001
    criterion = 'l1'
    input_type = 'map'
    output_type = 'coord'
    alias = 'Test'
    batch_size = 128
    shuffle = True
    optimizer = 'SGD'
    checkpoint_type = 'best'
    milestones = [50]
    gamma = 0.1
    checkpoint_dir = ''

@ex.capture
def get_trainer(parameters):
    return Trainer(parameters)


@ex.automain  # Using automain to enable command line integration.
def run():
    # Get params
    parameters = objdict(cfg())
    print(parameters)

    # Get the trainer object
    trainer = get_trainer(parameters)

    # Start a train
    trainer.run()

    # Start evaluation
    trainer.evaluation()
