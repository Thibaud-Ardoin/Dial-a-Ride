from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from icecream import ic

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
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ simple idx """
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
