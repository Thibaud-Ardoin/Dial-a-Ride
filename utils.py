from torch.cuda import is_available
import torch

def get_device():
    if is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(' - Device: ', device, ' - ')
    return device


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

def label2heatmap(labels, size):
    maps = torch.zeros((len(labels), size, size,)).type(torch.LongTensor)

    for i,label in enumerate(labels):
        maps[i][label[0]][label[1]] = 1
        # maps[i] = maps[i].view(maps[i].size(0), -1)
    return torch.flatten(maps, start_dim=1)
