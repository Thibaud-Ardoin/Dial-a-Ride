from torch.utils.data import ConcatDataset
import torch

dataset_dir = '/home/tibo/Documents/Prog/EPFL/own/data/supervision_data/'
files = ['s25000_t16_d2_i10_tlessFalse_funrf.pt', 's50000_t16_d2_i10_tlessFalse_funrf_version1.pt']
datasets = []

print('Concatenating the following datasets:', files)

outfile_name = input('What should the new dataset name be ?')
if outfile_name == '':
    outfile_name = 'concat.out'

for file in files :
    dataset = torch.load(dataset_dir + file)
    datasets.append(dataset)

all_data = ConcatDataset(datasets)

torch.save(all_data, dataset_dir + outfile_name)

print('Concatenation of all data has been savd as: ', dataset_dir + outfile_name)
