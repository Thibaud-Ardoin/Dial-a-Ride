from sacred import cli_option, Experiment
from sacred.observers import FileStorageObserver, TinyDbObserver, MongoObserver

from utils import objdict
from train import Trainer
from generator import PixelInstance


ex = Experiment("svm")

ex.observers.append(FileStorageObserver("my_runs"))
# ex.observers.append(MongoObserver(db_name='experiments_db'))


@ex.config  # Configuration is defined through local variables.
def cfg():
    epochs= 1
    model = 'UpAE'
    scheduler = 'plateau'
    data = './data/instances/split3_1nn_0k_n1_s50'
    patience = 100
    lr = 0.001
    criterion = 'crossentropy'
    input_type = 'map'
    output_type = 'map'
    alias = 'Test'
    batch_size = 128
    shuffle = True
    optimizer = 'SGD'
    checkpoint_type = 'best'
    milestones = [50]
    gamma = 0.1
    checkpoint_dir = ''
    layers = 256

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
