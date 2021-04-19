""" File defining the data instances """

import  numpy as np
import logging
import random
import time
import matplotlib.pyplot as plt
import os

import pickle

from dialRL.utils import image_coordonates2indices, indice2image_coordonates, distance, instance2Image_rep
from dialRL.dataset import tabu_parse, Driver, Target, tabu_parse_info



class DarPInstance():
    """ 2 Dimentional insance of the NN problem
    """
    def __init__(self, size, population, drivers, extremas=None, depot_position=None, time_end=1400, verbose=False):
        # Ground definition
        self.size = size
        self.nb_targets = population
        self.nb_drivers = drivers
        self.verbose = verbose
        self.time_end = time_end
        self.extremas = extremas

        # 2nd grade attributes
        self.drivers = []
        self.targets = []
        self.depot_position = depot_position


    def equal(self, x, y):
        return x[0] == y[0] and x[1] == y[1]

    def random_point(self):
        if self.extremas is None :
            # In case you need only integer points..
            pt = np.random.randint(0, self.size, (2))
        else :
            x = np.random.uniform(self.extremas[0], self.extremas[2])
            y = np.random.uniform(self.extremas[1], self.extremas[3])
            pt = np.array((x, y))
        return pt


    def random_generation(self, timeless=False, seed=None):
        """ Basicly populating the instance
        """

        if seed:
            np.random.seed(seed)

        # Generate Random points for targets and drivers
        if self.extremas is None :
            distinct_pts = np.random.choice(self.size**2, size=2*self.nb_targets + self.nb_drivers, replace=False)
            coordonates = [indice2image_coordonates(distinct_pts[i], self.size) for i in range(len(distinct_pts))]
        else :
            coordonates = [self.random_point() for i in range(2*self.nb_targets + self.nb_drivers)]

        # Populate Drivers
        for j in range(self.nb_drivers):
            if self.depot_position is None :
                driver = Driver(position=coordonates[j],
                                identity=j+1)
            else :
                driver = Driver(position=self.depot_position,
                                identity=j+1)
            self.drivers.append(driver)

        # Populate Targets
        for j in range(self.nb_targets):
            pickup = coordonates[self.nb_drivers + 2*j]
            dropoff = coordonates[self.nb_drivers + 2*j + 1]

            if timeless :
                tp1, tp2 = 0, self.time_end
            else :
                # Generate 50% of free dropof conditions, and 50% of free pickup time conditions
                intermediate_margin = 20
                tp1 = np.random.randint(intermediate_margin, self.time_end - intermediate_margin*2 - 1)
                tp2 = np.random.randint(tp1 + intermediate_margin, self.time_end - intermediate_margin)

            if j > self.nb_targets // 2 :
                start_fork = (0, self.time_end)
                end_fork = (tp1, tp2)
            else :
                start_fork = (tp1, tp2)
                end_fork = (0, self.time_end)

            target = Target(pickup, dropoff, start_fork, end_fork,
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

        if self.extremas is None:
            to_show = np.zeros((self.size, self.size))
            for driver in instance.drivers:
                to_show[driver.position[0]][driver.position[1]] = 3

            for target in instance.targets:
                to_show[target.pickup[0]][target.pickup[1]] = 1
                to_show[target.dropoff[0]][target.dropoff[1]] = 2

            plt.imshow(to_show)
            plt.show()

        else :
            image = instance2Image_rep(self.targets, self.drivers, self.size, time_step=self.time_step)
            plt.imshow(image)
            plt.show()
            exit()

if __name__ == '__main__':
    while 1 :
        data = './data/instances/cordeau2003/tabu1.txt'
        extremas, target_population, driver_population, time_end, depot_position, size = tabu_parse_info(data)
        # instance = DarPInstance(size=500, population=10, drivers=2, time_end=1400, verbose=True)
        instance = DarPInstance(size=size,
                                population=target_population,
                                drivers=driver_population,
                                depot_position=depot_position,
                                extremas=extremas,
                                time_end=time_end,
                                verbose=True)
        instance.random_generation()
        instance.reveal()
