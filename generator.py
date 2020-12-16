""" Define the Generator object """

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
    def __init__(self, size, population, verbose=False):

        # Ground definition
        self.size = size
        self.population = population
        self.center = ((self.size-1) / 2, (self.size-1) / 2)
        self.verbose = verbose

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

        self.image = np.zeros((self.size, self.size))

        # Generate Random points on the image
        self.points = [self.random_point()]
        while len(self.points) < self.population :
            point = self.random_point()
            if not np.sum([(point==self.points[i]).all() for i in range(len(self.points))]) :
                self.points.append(point)

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

        plt.imshow(self.image)
        plt.show()



class Generator:
    """ Object used to generate the needed data
        For example 2D pixel image for input NN problem
    """
    def __init__(self):
        return None


    def get_pixel_instance(self, size=10, population=3):
        instance=PixelInstance(size, population)
        instance.random_generation()
        return instance


    def generate_pixel_instances(self, size, population, number, save_type='numpy', unique_nn=False):
        instance_collection = []
        start_t = time.time()

        while len(instance_collection) < number :
            if save_type == 'pickle' :
                instance = self.get_pixel_instance(size, population)
                instance.random_generation()

                # Cast if we need a single nn
                if unique_nn and (len(instance.nearest_neighbors) == 1) :
                    instance_collection.append(instance)
                elif not unique_nn :
                    instance_collection.append(instance)

            elif save_type == 'numpy' :
                instance = self.get_pixel_instance(size, population)
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
    # Params :
    save_type = 'pickle'
    size_of_images = 50
    number_of_pixel_per_image = 1
    size_of_data = 25000
    unique_nn = True
    instances_name = 'split3_{0}nn_{1}k_n{2}_s{3}'.format(int(unique_nn), size_of_data//1000, number_of_pixel_per_image, size_of_images)


    if os.path.isdir('./data/instances/' + instances_name) :
        raise "Folder is already in place"
    else :
        os.mkdir('./data/instances/' + instances_name)

    generator = Generator()
    for name in ['test_instances', 'train_instances', 'validation_instances'] :
        instances = generator.generate_pixel_instances(size=size_of_images,
                                                       population=number_of_pixel_per_image,
                                                       number=size_of_data,
                                                       save_type=save_type,
                                                       unique_nn=unique_nn)
        generator.save_instances(instances,
                                 path='./data/instances/' + instances_name,
                                 save_type=save_type,
                                 name=name)
