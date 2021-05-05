import torch
from icecream import ic
import matplotlib.pyplot as plt

def printing(mat):
    # mat = positional_encoding(max_pos, d_model, 0)
    ic(mat.shape)
    d_model = mat.shape[1]
    plt.pcolormesh(mat, cmap='copper')
    plt.xlabel('Depth')
    plt.xlim((0, d_model))
    plt.ylabel('Position')
    plt.title("PE matrix heat map")
    plt.colorbar()
    plt.show()


def generate_positional_encoding(d_model, max_len):
    """
    Create standard transformer PEs.
    Inputs :
      d_model is a scalar correspoding to the hidden dimension
      max_len is the maximum length of the sequence
    Output :
      pe of size (max_len, d_model), where d_model=dim_emb, max_len=1000
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:,0::2] = torch.sin(position * div_term)
    pe[:,1::2] = torch.cos(position * div_term)
    return pe

x = [[0,0], [1,1]]
nb_nodes = len(x)
bsz = 1
dim_emb = 128

PE = generate_positional_encoding(dim_emb, 1000)

ic(PE[0])
ic(PE[1])
ic(PE[2])
ic(PE.shape)

printing(PE.numpy())

# ic(torch.nn.Parameter(torch.randn(dim_emb)))
# ic(torch.nn.Parameter(torch.randn(dim_emb)).shape)
#
# ic(PE[1].repeat(bsz,1))
# ic(PE[1].repeat(bsz,1).shape)

idx_start_placeholder = torch.Tensor([nb_nodes]).long().repeat(bsz)
ic(idx_start_placeholder)
h_start = h_encoder[zero_to_bsz, idx_start_placeholder, :] + self.PE[0].repeat(bsz,1) # size(h_start)=(bsz, dim_emb)
