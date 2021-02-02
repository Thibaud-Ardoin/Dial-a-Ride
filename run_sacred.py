import sys
from sacred import cli_option, Experiment
from sacred.observers import FileStorageObserver, TinyDbObserver, MongoObserver
import argparse

from utils import objdict
from train import Trainer
from instances import PixelInstance
import models


ex = Experiment("svm")

ex.observers.append(FileStorageObserver("experiments_obs"))
# ex.observers.append(MongoObserver(db_name='mongodb://127.0.0.1:27017/database') #'experiments_db')

@ex.capture
def get_trainer(parameters, sacred):
    return Trainer(parameters, sacred)

ex.add_config(
    epochs= 2000,
    model = 'Trans1',
    scheduler = 'plateau',
    data = './data/instances/split3_1nn_10k_n2_s50_m1_t1',
    patience = 10,
    lr = 0.001,
    criterion = 'crossentropy',
    input_type = 'flatmap',
    output_type = 'flatmap',
    alias = 'test_transformer',
    batch_size = 128,
    shuffle = True,
    optimizer = 'Adam',
    checkpoint_type = 'best',
    milestones = [50],
    gamma = 0.1,
    checkpoint_dir = '',
    layers = 256,
    channels = 1,
    file_dir = '/experiments_obs/',
    upscale_factor = 2
)


@ex.automain  # Using automain to enable command line integration.
def run(_run):
    # Get params
    parameters = objdict(_run.config)

    # Get the trainer object
    trainer = get_trainer(parameters, sacred=_run)

    # Start a train
    trainer.run()

    # Start evaluation
    trainer.evaluation()
