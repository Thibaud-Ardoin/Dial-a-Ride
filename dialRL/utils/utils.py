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

def heatmap2image_coord(heatmap):
    coord = np.where(heatmap == heatmap.max())
    return coord

def label2heatmap(labels, size):
    maps = torch.zeros((len(labels), size, size,)).type(torch.LongTensor)

    for i,label in enumerate(labels):
        maps[i][label[0]][label[1]] = 1
        # maps[i] = maps[i].view(maps[i].size(0), -1)
    # visualize(maps[0], txt='first elmt of the map of label2heatmap (before torch.flatten)')
    return torch.flatten(maps, start_dim=1)


def indices2image(indice_list, image_size):
    indice_map = torch.zeros(image_size**2)
    for i, indice in enumerate(indice_list):
        indice_map[indice.item()] = 1.
        if i==(len(indice_list) - 1):
            indice_map[indice.item()] = 0.5
    return indice_map2image(indice_map, image_size)

def instance2world(indice_list, type_vector, image_size):
    indice_map = torch.zeros(image_size**2)
    for i, indice in enumerate(indice_list):
        indice_map[indice] = type_vector[i]
    return indice_map2image(indice_map, image_size)


def image_coordonates2indices(coord, image_size):
    x, y = coord
    # Row then collon ts. Id = x*image_size + y
    # Going from 0 to iamge_size^2 -1 = (image_size - 1)*image_size + image_size - 1
    return x*image_size + y


def indice2image_coordonates(indice, image_size):
    # x = Id // image_size ; y = Id%image_size
    return indice // image_size, indice%image_size


def indice_map2image(indice_map, image_size):
    # x = Id // image_size ; y = Id%image_size
    return torch.reshape(indice_map, (image_size, image_size))

def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


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
