import numpy as np
import matplotlib.pyplot as plt
import torch
from icecream import ic

def printing(mat):
    # mat = positional_encoding(max_pos, d_model, 0)
    ic(mat.shape)
    plt.pcolormesh(mat, cmap='copper')
    plt.xlabel('Depth')
    plt.xlim((0, d_model))
    plt.ylabel('Position')
    plt.title("PE matrix heat map")
    plt.colorbar()
    plt.show()



def fourier_feature(coordonates, B_gauss):
    # coordonates = torch.stack(coordonates).permute(1, 0)
    pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2 * 2)
    x = (pi * coordonates).double()
    transB = torch.transpose(B_gauss, 0, 1).double()
    if x.shape[1] == 4:
        transB = torch.cat([transB, transB])
    x_proj = x.matmul(transB)
    final = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
    return final


#### TensorFlow only version ####
def positional_encoding(max_position, d_model, min_freq=1e-4):
    position = tf.range(max_position, dtype=tf.float32)
    mask = tf.range(d_model)
    sin_mask = tf.cast(mask%2, tf.float32)
    cos_mask = 1-sin_mask
    exponent = 2*(mask//2)
    exponent = tf.cast(exponent, tf.float32)/tf.cast(d_model, tf.float32)
    freqs = min_freq**exponent
    angles = tf.einsum('i,j->ij', position, freqs)
    pos_enc = tf.math.cos(angles)*cos_mask + tf.math.sin(angles)*sin_mask
    return pos_enc

#### Numpy version ####
def positional_encoding(max_position, d_model, y, min_freq=1e-4):
    position = np.arange(max_position)
    Yposition = np.full((max_position), y)
    freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
    Ypos_enc = Yposition.reshape(-1,1)*freqs.reshape(1,-1)
    print(np.shape(pos_enc))
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
    return pos_enc

### Plotting ####
scale = 1000
d_model = 64
max_pos = 64

mapping_size =  d_model
B_gauss = torch.normal(0, 1, size=(mapping_size, 2)) * scale
# B_gauss = torch.eye(2)
# B_gauss = torch.normal(0, 1, (mapping_size, 2))



ffs = []
x = 0
for y in range(200):
    a, b =np.random.random(), np.random.random()
    ffs.append(fourier_feature(torch.tensor([[a, b]]), B_gauss).squeeze().numpy())

# a, b =np.random.random(), np.random.random()
for y in range(200):
    ffs.append(fourier_feature(torch.tensor([[y/200, y/200]]), B_gauss).squeeze().numpy())


# ic(ff)
printing(np.array(ffs))
