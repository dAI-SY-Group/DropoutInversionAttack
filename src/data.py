import os

import torch

def load_victim_data(victim_data_path, batch_size):
    victim_data_path = victim_data_path if victim_data_path.endswith('.tdump') else victim_data_path
    if os.path.exists(victim_data_path):
        vic_data = torch.load(victim_data_path)
        print(f'Loaded victim data from {victim_data_path}!')
        data_shape = vic_data['inputs'][0].shape 
        return list(vic_data['inputs'].split(batch_size)), list(vic_data['targets'].split(batch_size)), data_shape
    else:
        raise ValueError(f'Tried loading victim data from {victim_data_path}, but that does not exist. Try another path and make sure this victim dataset was already generated!')
