from torch.cuda import is_available
import torch
import matplotlib.pyplot as plt
import numpy as np

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
    # visualize(maps[0], txt='first elmt of the map of label2heatmap (before torch.flatten)')
    return torch.flatten(maps, start_dim=1)

def visualize(image, txt=''):
    if len(image.shape) == 3 and image.shape[0] in [1,2,3]:
        img = image.cpu().numpy()
        img = np.squeeze(np.transpose(img, (1,2,0)))
    elif not type(image) == np.ndarray:
        img = image.cpu().numpy()
    else :
        img = image.copy()

    print('\n ** Visualizer - Shape of image: ', img.shape, '\n' + txt + '\n')
    print('Argmax : ', np.argmax(img))
    print('min and max : ', np.min(img), np.max(img))
    print(img)
    if len(img.shape)>2 and img.shape[2]==2:
        img = np.stack((img[:,:,0], img[:,:,0], img[:,:,1]), axis=2)
    if len(img.shape)>1 :
        plt.imshow(img)
        plt.show()
