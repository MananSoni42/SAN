'''
The main classifier Module
Some parts of this module have been taken from the author
'''

import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from .dropout_wrapper import DropoutWrapper

class Classifier(torch.nn.Module):
    def __init__(self, x_size, y_size, opt, prefix='decoder', dropout=None):
        super(Classifier, self).__init__()
        self.opt = opt
        if dropout is None: # apply dropout (passing p=0 is equivalent to no dropout)
            self.dropout = DropoutWrapper(opt.get(f'{prefix}_dropout_p', 0))
        else:
            self.dropout = dropout

        # if set the data [x, y] will be augmented to [x, y, |x-y|, x*y]
        self.merge_opt = opt.get(f'{prefix}_merge_opt', 0)

        self.weight_norm_on = opt.get(f'{prefix}_weight_norm_on', False)

        if self.merge_opt == 1:
            self.proj = torch.nn.Linear(x_size * 4, y_size)
        else:
            self.proj = torch.nn.Linear(x_size * 2, y_size)

        if self.weight_norm_on:
            self.proj = torch.nn.utils.weight_norm(self.proj) # apply weight normalization

    def forward(self, x1, x2, mask=None):
        if self.merge_opt == 1:
            x = torch.cat([x1, x2, (x1 - x2).abs(), x1 * x2], 1)
        else:
            x = torch.cat([x1, x2], 1)
        x = self.dropout(x)
        scores = self.proj(x) # results of the linear layer (with or without augmentation)
        return scores
