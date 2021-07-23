from torch.cuda import is_available
import torch
import matplotlib.pyplot as plt
import numpy as np
import drawSvg as draw
import tempfile
from icecream import ic
import time
import math

def get_device():
    if False: #is_available(): #False: #
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(' - Device: ', device, ' - ')
    return device

def trans25_coord2int(coord, src_vocab_size, extremas):
    siderange = int(math.sqrt(src_vocab_size))
    boxh, boxw = abs(extremas[2] - extremas[0]) / siderange, abs(extremas[3] - extremas[1]) / siderange
    h, w = abs(coordonate[:,0] - extremas[0]) / boxh, abs(coordonate[:,1] -  extremas[1]) / boxw
    return h.add(w * siderange).long()

def quinconx(l, d=1):
    nb = len(l)
    if nb==2:
        a, b = l
        return torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1).flatten(start_dim=d)
    elif nb==3:
        a, b, c = l
        q1 = torch.cat([b.unsqueeze(-1), c.unsqueeze(-1)], dim=-1).flatten(start_dim=d)
        return torch.cat([a.unsqueeze(-1), q1.unsqueeze(-1)], dim=-1).flatten(start_dim=d)
    elif nb==4:
        a, b, c, dd = l
        return torch.cat([a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1), dd.unsqueeze(-1)], dim=-1).flatten(start_dim=d)

def norm_image(self, image, type=None, scale=1):
    image = np.kron(image, np.ones((scale, scale)))
    if type=='rgb':
        ret = np.empty((image.shape[0], image.shape[0], 3), dtype=np.uint8)
        ret[:, :, 0] = image.copy()
        ret[:, :, 1] = image.copy()
        ret[:, :, 2] = image.copy()
        image = ret.copy()
    return (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)


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
    indice_map = np.zeros(image_size**2)
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


def coord2int(coord):
    number_precision = 3
    new_coord = int(round(coord, number_precision)*(10**number_precision))
    return new_coord

def time2int(time):
    number_precision = 0
    new_coord = int(round(time, number_precision)*(10**number_precision))
    return new_coord


def obs2int(obs):
    mi = -int(min(obs))
    new_obs = [int(m) + mi for m in obs]
    return new_obs


def indice_map2image(indice_map, image_size):
    # x = Id // image_size ; y = Id%image_size
    return np.reshape(indice_map, (image_size, image_size))

def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def float_equality(f1, f2, eps=0.001):
    return abs(f1 - f2) < eps


def GAP_function(cost, best_cost):
    if best_cost is None :
        return None
    plus = cost - best_cost
    return 100 * plus / best_cost


def plotting(mat):
    d_model = mat.shape[-1]
    ic(d_model)
    ic(mat)
    # mat = positional_encoding(max_pos, d_model, 0)
    plt.pcolormesh(np.array(mat[0]), cmap='copper')
    plt.xlabel('Depth')
    plt.xlim((0, d_model))
    plt.ylabel('Position')
    plt.title("PE matrix heat map")
    plt.colorbar()
    plt.show()
    time.sleep(5)


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
