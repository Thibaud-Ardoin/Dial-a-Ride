""" File defining the data instances """

import  numpy as np
import logging
import random
import time
import matplotlib.pyplot as plt
import os

import pickle

from utils import image_coordonates2indices, indice2image_coordonates

class PixelInstance():
    """ 2 Dimentional insance of the NN problem
    """
    def __init__(self, size, population, drivers, moving_car, transformer, verbose=False):

        # Ground definition
        self.size = size
        self.population = population
        self.verbose = verbose
        self.moving_car = moving_car
        self.transformer = transformer
        self.drivers = drivers

        # 2nd grade attributes
        self.points = []
        self.centers = []
        self.caracteristics = None
        self.image = None
        self.neighbors_lists = None
        self.distances_lists = None
        self.type_vactor = None

        if self.moving_car or self.transformer:
            for _ in range(self.drivers) :
                self.centers.append(self.random_point())
        else :
            self.centers = [((self.size-1) / 2, (self.size-1) / 2)]

    def equal(self, x, y):
        return x[0] == y[0] and x[1] == y[1]

    def random_point(self):
        # Distinct selection
        done = False
        while not done:
            pt = np.random.randint(0, self.size, (2))
            if not np.sum([self.equal(pt, self.points[i]) for i in range(len(self.points))] + [self.equal(pt, self.centers[i]) for i in range(len(self.centers))]) :
                done = True
        return pt


    def random_generation(self, seed=None):
        """ Basicly populating the instance
            Using distance to center
        """

        if seed:
            np.random.seed(seed)

        # Generate Random points (distinct) on the image
        for _ in range(self.population):
            self.points.append(self.random_point())

        if self.moving_car and not self.transformer:
            self.image = np.zeros((self.size, self.size, 2))
            # Set coordonates according to channel_type
            for point in self.points :
                self.image[point[0], point[1], 0] = 1
            self.image[self.center[0], self.center[1], 1] = 1

        if self.transformer:
            self.image = [image_coordonates2indices(self.points[i], self.size) for i in range(len(self.points))]
            self.image = self.image + [image_coordonates2indices(self.centers[i], self.size) for i in range(self.drivers)]

            # -1 as marker for targets and i for drivers. 0 beeing for neutral space
            self.caracteristics = [-1 for _ in range(self.population)] + [i+1 for i in range(self.drivers)]

        else :
            self.image = np.zeros((self.size, self.size))
            # Set image coordonates of the points to 1
            for point in self.points :
                self.image[point[0], point[1]] = 1

        # List of the distances, ordered as self.points is (Creation order)
        self.distances_lists = [list(map(lambda x: np.linalg.norm(self.centers[i] - x), self.points)) for i in range(self.drivers)]
        # List of points ordered by the distance (smallest distance first)
        self.neighbors_lists = [[x for _,x in sorted(zip(self.distances_lists[i],self.points), key=lambda x: x[0])] for i in range(self.drivers)]

        # nn_count = sum(1 for dist in self.distance_list if dist == np.min(self.distance_list))
        # List of the points  at smalles distance
        # self.nearest_neighbors = [self.neighbor_list[i] for i in range(nn_count)]

        if self.verbose:
            print('Random generation  concluded')


    def reveal(self):
        for item in vars(self):
            print(item, ':', vars(self)[item])

        to_show = self.image
        if self.moving_car and not self.transformer :
            to_show = np.append(self.image, [self.image[0]], axis=0)
            to_show = np.stack((to_show[:,:,0], to_show[:,:,0], to_show[:,:,1]), axis=2)
            # to_show[self.nearest_neighbors[0][0], self.nearest_neighbors[0][1], 1] = 0.5
            # to_show = np.transpose(to_show, (1, 2, 0))
        # else :
            # to_show[self.nearest_neighbors[0][0], self.nearest_neighbors[0][1]] = 0.5
        plt.imshow(to_show)
        plt.show()


if __name__ == '__main__':
    while 1 :
        instance = PixelInstance(size=50, population=10, drivers=2, moving_car=True, transformer=True, verbose=True)
        instance.random_generation()
        print(instance.random_point())
        instance.reveal()
