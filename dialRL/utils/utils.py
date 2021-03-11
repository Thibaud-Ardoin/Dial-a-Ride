from torch.cuda import is_available
import torch
import matplotlib.pyplot as plt
import numpy as np
import drawSvg as draw
import tempfile
import matplotlib.pyplot as plt

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


def indice_map2image(indice_map, image_size):
    # x = Id // image_size ; y = Id%image_size
    return np.reshape(indice_map, (image_size, image_size))

def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

colors_list = [
    '#faff18',
    '#ff18dd',
    '#a818ff',
    '#3318ff',
    '#18aeff',
    '#18ffce',
    '#18ff59',
    '#78ff18',
    '#ff7a18',
    '#ff1818',
    '#ff186c',
    '#25731b',
    '#73731b',
    '#1b7367',
    '#1b4a73',
    '#4f1b73',
    '#1b7351',
]

def instance2Image_rep(targets, drivers, size):
    # Return an image gathered from svg data

    d = draw.Drawing(size*2, size*2, origin='center', displayInline=False)
    for target in targets:
        # draw two nodes for pickup and delivery + An arrow connecting them
        if target.state > -1 :
            d.append(draw.Circle(target.pickup[0],target.pickup[1], 0.2,
                fill='#8b5962', stroke_width=0.05, stroke='#555455'))
        else :
            d.append(draw.Circle(target.pickup[0],target.pickup[1], 0.2,
                    fill='red', stroke_width=0.05, stroke='black'))

        if target.state > 0 :
            d.append(draw.Circle(target.dropoff[0],target.dropoff[1], 0.2,
                fill='#59598b', stroke_width=0.05, stroke='#555455'))
            d.append(draw.Line(target.pickup[0],target.pickup[1],
                               target.dropoff[0],target.dropoff[1],
                               stroke='#516e4a', stroke_width=0.01, fill='none'))
        else :
            d.append(draw.Circle(target.dropoff[0],target.dropoff[1], 0.2,
                    fill='blue', stroke_width=0.05, stroke='black'))
            d.append(draw.Line(target.pickup[0],target.pickup[1],
                               target.dropoff[0],target.dropoff[1],
                               stroke='green', stroke_width=0.02, fill='none'))

    for driver in drivers :
        d.append(draw.Circle(driver.position[0], driver.position[1], 0.3,
                fill=colors_list[driver.identity - 1], stroke_width=0.2, stroke='black'))

    #d.setPixelScale(2)  # Set number of pixels per geometry unit
    d.setRenderSize(size*40, size*40)
    fo = tempfile.NamedTemporaryFile()
    d.savePng(fo.name)
    array_image = np.array(plt.imread(fo.name))
    fo.close()
    return array_image


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
