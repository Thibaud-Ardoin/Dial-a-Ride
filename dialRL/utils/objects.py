from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from icecream import ic
import torch
import numpy as np

class MemoryDataset(Dataset):
    """ Customed Dataset class for our Instances data
    """
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        """ Returns a couple (image, neares 1hot)"""
        instance = self.data[idx]
        return instance

class SupervisionDataset(Dataset):
    """ Customed Dataset class for our Instances data
    """
    def __init__(self, data_list, typ=None, augment=None):
        # super(SupervisionDataset, self).__init__(data_list)
        self.data = data_list
        self.augment = augment
        self.typ = typ


    def __len__(self):
        return len(self.data)


    def pos_augmentation(self, positions):
        ''' Transposing all popsition information by a 2D vectors in [-1,1]²
            In addition the 2D points are rotated by a random value [-pi, pi]
        '''
        depot, targets, drivers = positions

        get_rot = lambda theta: torch.tensor([[torch.cos(theta*np.pi), -torch.sin(theta*np.pi)],
                                [torch.sin(theta*np.pi), torch.cos(theta*np.pi)]]).double()

        rand_vect = torch.rand(2 + 1) * 2 - 1
        rot = get_rot(torch.tensor(0))#rand_vect[-1])
        trsp = rand_vect[:-1]

        comp_aug2d = lambda pos: torch.matmul(torch.from_numpy(pos), rot) + trsp
        comp_aug4d = lambda pos: torch.cat([torch.matmul(torch.from_numpy(pos[:2]), rot) + trsp,
                                            torch.matmul(torch.from_numpy(pos[2:]), rot) + trsp])

        depot = comp_aug2d(depot)
        targets = list(map(comp_aug4d, targets))
        drivers = list(map(comp_aug2d, drivers))

        return [depot, targets, drivers]


    def time_augmentation(self, time):
        ''' Shifting all time information by a random amount
            This transposition value is in [0, 10]
            half of the time it's rounded to an integer
        '''
        current, targets, drivers = time

        rand1, rand2 = torch.rand(2)
        trsp = rand1 * 10
        if rand2 > 0.5:
            trsp = torch.round(trsp)

        aug4d = lambda tem: torch.from_numpy(tem) + trsp
        aug1d = lambda tem: tem + trsp

        current = aug1d(current)
        targets = list(map(aug4d, targets))
        drivers = list(map(aug1d, drivers))

        return [current, targets, drivers]


    def __getitem__(self, idx):
        """ simple idx """
        obs, sup = self.data[idx]
        world, targets, drivers, positions, time_constraints = obs
        if hasattr(self, 'augment') and self.augment is not None and self.augment:
            positions = self.pos_augmentation(positions)
            time_constraints = self.time_augmentation(time_constraints)

        return self.data[idx]

class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)
