'''
List all possibilities of linear functions and various weight initializations
For general use anywhere in the code
'''

import torch

def linear(x):
    return x

act_poss = { # List all possibilities of various activations
    'tanh': torch.nn.functional.tanh,
    'relu': torch.nn.functional.relu,
    'prelu': torch.nn.functional.prelu,
    'leaky_relu': torch.nn.functional.leaky_relu,
    'sigmoid': torch.nn.functional.sigmoid,
    'elu': torch.nn.functional.elu,
    'selu': torch.nn.functional.selu,
}

def activation(func_name):
    return act_poss[func_name]

init_poss = { # List of all possibilites weight initiations
    'uniform': torch.nn.init.uniform,
    'normal': torch.nn.init.normal,
    'eye': torch.nn.init.eye,
    'xavier_uniform': torch.nn.init.xavier_uniform,
    'xavier_normal': torch.nn.init.xavier_normal,
    'kaiming_uniform': torch.nn.init.kaiming_uniform,
    'kaiming_normal': torch.nn.init.kaiming_normal,
    'orthogonal': torch.nn.init.orthogonal,
}

def init_wrapper(init_name='xavier_uniform'):
    return init_poss[init_name]
