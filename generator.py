""" Define the Generator object """

import  numpy as np
import logging
import random
import time
import matplotlib.pyplot as plt
import os
import sys
import argparse

import pickle

from instances import PixelInstance

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Generator.",
        epilog="python train.py --param X")

    # required input parameters
    parser.add_argument(
        '--save_type', type=str,  default='pickle')
    parser.add_argument(
        '--size_of_images', type=int,  default=50)
    parser.add_argument(
        '--number_of_pixel_per_image', type=int,  default=2)
    parser.add_argument(
        '--size_of_data', type=int,  default=500000)
    parser.add_argument(
        '--unique_nn', type=bool,  default=True)
    parser.add_argument(
        '--moving_car', type=bool, default=True)
    parser.add_argument(
        '--codding', type=str, default='2channels')
    return parser.parse_known_args(args)[0]


class Generator:
    """ Object used to generate the needed data
        For example 2D pixel image for input NN problem
    """
    def __init__(self, size, population, moving_car, codding='2channels'):
        self.size = size
        self.population = population
        self.moving_car = True
        self.codding = '2channels'


    def get_pixel_instance(self):
        instance=PixelInstance(self.size, self.population, self.moving_car, codding=self.codding)
        instance.random_generation()
        return instance


    def generate_pixel_instances(self, number, save_type='numpy', unique_nn=False):
        instance_collection = []
        start_t = time.time()

        while len(instance_collection) < number :
            if save_type == 'pickle' :
                instance = self.get_pixel_instance()
                instance.random_generation()

                # Cast if we need a single nn
                if unique_nn and (len(instance.nearest_neighbors) == 1) :
                    instance_collection.append(instance)
                elif not unique_nn :
                    instance_collection.append(instance)

            elif save_type == 'numpy' :
                instance = self.get_pixel_instance()
                instance.random_generation()
                if unique_nn and (len(instance.nearest_neighbors) == 1) :
                    instance_collection.append([instance.image, instance.neighbor_list])
                elif not unique_nn :
                    instance_collection.append([instance.image, instance.neighbor_list])

        print(' - Done with Instances Generation - ')
        print('\t *', number, ' instances in ', time.time() - start_t, 'sec')
        return instance_collection


    def save_instances(self, instances, path, name, save_type='numpy'):
        if name == '':
            name = 'instances_s=' + str(len(instances)) + '_t=' + time.strftime("%d-%H-%M")
        if save_type == 'pickle' :
            name = name + '.pkl'
            filehandler = open('/'.join([path, name]), 'wb')
            pickle.dump(instances, filehandler)
            filehandler.close()
        elif save_type == 'numpy':
            name = name + '.npy'
            with open('/'.join([path, name]), 'wb') as f:
                np.save(f, np.array(instances))
                f.close()
        print(' - Done with Instances Saving - ')
        print('\t *', 'saved as: ', name)


if __name__ == '__main__':
    parameters = parse_args(sys.argv[1:])

    # Params :

    instances_name = 'split3_{0}nn_{1}k_n{2}_s{3}_m{4}'.format(int(parameters.unique_nn),
                                                          parameters.size_of_data//1000,
                                                          parameters.number_of_pixel_per_image,
                                                          parameters.size_of_images,
                                                          int(parameters.moving_car))

    if os.path.isdir('./data/instances/' + instances_name) :
        raise "Folder is already in place"
    else :
        os.mkdir('./data/instances/' + instances_name)

    generator = Generator(size=parameters.size_of_images,
                          population=parameters.number_of_pixel_per_image,
                          moving_car=parameters.moving_car,
                          codding=parameters.codding)

    for name in ['test_instances', 'train_instances', 'validation_instances'] :
        instances = generator.generate_pixel_instances(number=parameters.size_of_data,
                                                       save_type=parameters.save_type,
                                                       unique_nn=parameters.unique_nn)
        generator.save_instances(instances,
                                 path='./data/instances/' + instances_name,
                                 save_type=parameters.save_type,
                                 name=name)
