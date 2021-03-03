""" File defining the data instances """

import  numpy as np
import logging
import random
import time
import matplotlib.pyplot as plt
import os

import pickle

from dialRL.utils import image_coordonates2indices, indice2image_coordonates, distance

class Target():
    def __init__(self, pickup, dropoff, start, end, identity, weight=0):
        self.pickup = pickup
        self.dropoff = dropoff
        self.start = start
        self.end = end
        self.weight = weight
        self.identity = identity

        # State is in [-1, 0, 1] for [wait pick up, in a car, done]
        self.state = -1
        self.available = 0


class Driver():
    def __init__(self, position, max_capacity=2, speed=1, verbose=False):
        self.position = position
        self.max_capacity = max_capacity
        self.speed = speed
        self.distance = 0
        self.loaded = [] #Target list

    def move(self, new_position):
        self.distance = distance(new_position, self.position)
        self.position = new_position

    def capacity(self):
        c = 0
        for target in self.loaded:
            c += target.weight
        return c

    def load(self, target):
        if target.weight + self.capacity() > self.max_capacity :
            return False
        else :
            self.loaded.append(target)
            return True

    def unload(self, target):
        indice = target.identity
        for i,t in enumerate(self.loaded):
            if t.identity == indice:
                del self.loaded[i]
                return True
        return False


    def is_in(self, indice):
        for target in self.loaded:
            if target.identity == indice:
                return True
        return False


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
            driver = Driver(distincs_coordonates[j])
            self.drivers.append(driver)

        # Populate Targets
        for j in range(self.nb_targets):
            pickup = distincs_coordonates[self.nb_drivers + 2*j]
            dropoff = distincs_coordonates[self.nb_drivers + 2*j + 1]
            start = 0
            end = self.time_end
            target = Target(pickup, dropoff, start, end, identity=j+self.nb_drivers)
            self.targets.append(target)

        if self.verbose:
            print('Random generation  concluded')


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
        instance = DarPInstance(size=500, population=10, drivers=2,time_end=1400, verbose=True)
        instance.random_generation()
        instance.reveal()
