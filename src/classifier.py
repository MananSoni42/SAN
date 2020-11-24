'''
The main classifier Module
Some parts of this module have been taken from the author
'''

import torch
from torch.autograd import Variable
from .dropout_wrapper import DropoutWrapper

class Classifier(torch.nn.Module):
    def __init__(self, x_size, y_size, opt, prefix='decoder', dropout=None):
        super(Classifier, self).__init__()

        self.opt = opt # Store all the options

        # apply dropout (passing p=0 is equivalent to no dropout)
        self.dropout = DropoutWrapper(opt.get(f'{prefix}_dropout_p', 0)) if not dropout else dropout

        # if set the data [x, y] will be augmented to [x, y, |x-y|, x*y]
        self.merge_opt = opt.get(f'{prefix}_merge_opt', 0)

        x_aug_size = 4 if self.merge_opt else 2
        self.project = torch.nn.Linear(x_aug_size)

        # apply weight normalization
        if opt.get(f'{prefix}_weight_norm_on', False):
            self.proj = torch.nn.utils.weight_norm(self.proj)

    def forward(self, x1, x2, mask=None):
        x_inp = [x1, x2, (x1 - x2).abs(), x1 * x2] if self.merge_opt else [x1, x2]
        x = torch.cat(x_inp, 1)
        x = self.dropout(x)
        return self.proj(x)# results of the linear layer (with or without augmentation)
