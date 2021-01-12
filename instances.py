""" File defining the data instances """

import  numpy as np
import logging
import random
import time
import matplotlib.pyplot as plt
import os

import pickle

class PixelInstance():
    """ 2 Dimentional insance of the NN problem
    """
    def __init__(self, size, population, moving_car, codding='2channels', verbose=False):

        # Ground definition
        self.size = size
        self.population = population
        self.verbose = verbose
        self.codding = codding
        if moving_car :
            self.center = self.random_point()
        else :
            self.center = ((self.size-1) / 2, (self.size-1) / 2)

        # 2nd grade attributes
        self.image = None
        self.points = None
        self.neighbor_list = None
        self.distance_list = None


    def random_point(self):
        return np.random.randint(0, self.size, (2))


    def random_generation(self, seed=None):
        """ Basicly populating the instance
            Using distance to center
        """

        if seed:
            np.random.seed(seed)

        # Generate Random points (distinct) on the image
        self.points = [self.random_point()]
        while len(self.points) < self.population :
            point = self.random_point()
            if not np.sum([(point==self.points[i]).all() for i in range(len(self.points))]) :
                self.points.append(point)


        if self.codding=='2channels':
            self.image = np.zeros((self.size, self.size, 2))
            # Set coordonates according to channel_type
            for point in self.points :
                self.image[point[0], point[1], 0] = 1
            self.image[self.center[0], self.center[1], 1] = 1
        else :
            self.image = np.zeros((self.size, self.size))
            # Set image coordonates of the points to 1
            for point in self.points :
                self.image[point[0], point[1]] = 1

        # Calculate the nearest neighbors
        self.distance_list = list(map(lambda x: np.linalg.norm(self.center - x), self.points))
        self.neighbor_list = [x for _,x in sorted(zip(self.distance_list,self.points), key=lambda x: x[0])]

        nn_count = sum(1 for dist in self.distance_list if dist == np.min(self.distance_list))
        self.nearest_neighbors = [self.neighbor_list[i] for i in range(nn_count)]

        if self.verbose:
            print('Random generationo  concluded')


    def reveal(self):
        for item in vars(self):
            print(item, ':', vars(self)[item])

        to_show = self.image
        if self.codding == '2channels':
            print('need to concat coretly')
            to_show = np.append(self.image, [self.image[0]], axis=0)
            # to_show = np.transpose(to_show, (1, 2, 0))

        plt.imshow(to_show)
        plt.show()


if __name__ == '__main__':
    instance = PixelInstance(size=50, population=5, moving_car=True, codding='2channels', verbose=True)
    instance.random_generation()
    instance.reveal()
