""" File defining the data instances """

import  numpy as np
import logging
import random
import time
import matplotlib.pyplot as plt
import os

import pickle

from dialRL.utils import image_coordonates2indices, indice2image_coordonates, distance
from dialRL.dataset import tabu_parse, Driver, Target



class DarPInstance():
    """ 2 Dimentional insance of the NN problem
    """
    def __init__(self, size, population, drivers, time_end, verbose=False):
        # Ground definition
        self.size = size
        self.nb_targets = population
        self.nb_drivers = drivers
        self.verbose = verbose
        self.time_end = time_end

        # 2nd grade attributes
        self.drivers = []
        self.targets = []
        self.depot_position = None


    def equal(self, x, y):
        return x[0] == y[0] and x[1] == y[1]

    def random_point(self):
        pt = np.random.randint(0, self.size, (2))
        return pt


    def random_generation(self, seed=None):
        """ Basicly populating the instance
        """

        if seed:
            np.random.seed(seed)

        # Generate Random points for targets and drivers
        distinct_pts = np.random.choice(self.size**2, size=2*self.nb_targets + self.nb_drivers, replace=False)
        distincs_coordonates = [indice2image_coordonates(distinct_pts[i], self.size) for i in range(len(distinct_pts))]

        # Populate Drivers
        for j in range(self.nb_drivers):
            driver = Driver(position=distincs_coordonates[j],
                            identity=j+1)
            self.drivers.append(driver)

        # Populate Targets
        for j in range(self.nb_targets):
            pickup = distincs_coordonates[self.nb_drivers + 2*j]
            dropoff = distincs_coordonates[self.nb_drivers + 2*j + 1]
            start = (0, 0)
            end = (self.time_end, self.time_end)
            target = Target(pickup, dropoff, start, end,
                            identity=j + 1)
            self.targets.append(target)

        if self.verbose:
            print('Random generation  concluded')


    def dataset_generation(self, data_name):
        """ Basicly populating the instance
        """
        targets, drivers = tabu_parse(data_name)
        self.depot_position = drivers[0].position
        self.drivers = drivers
        self.targets = targets

        if self.verbose:
            print('Dataset loaded as DARP instance')


    def reveal(self):
        for item in vars(self):
            print(item, ':', vars(self)[item])

        to_show = np.zeros((self.size, self.size))
        for driver in instance.drivers:
            to_show[driver.position[0]][driver.position[1]] = 3

        for target in instance.targets:
            to_show[target.pickup[0]][target.pickup[1]] = 1
            to_show[target.dropoff[0]][target.dropoff[1]] = 2

        plt.imshow(to_show)
        plt.show()


if __name__ == '__main__':
    while 1 :
        instance = DarPInstance(size=500, population=10, drivers=2, time_end=1400, verbose=True)
        instance.dataset_generation('./data/instances/cordeau2003/tabu1.txt')
        instance.reveal()
