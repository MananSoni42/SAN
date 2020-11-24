'''
Generic dropout wrapper
Can be wrapped around any torch layer
'''

import torch
from torch.autograd import Variable

class DropoutWrapper(torch.nn.Module):
    ''' Aply this wrapper to enable dropout on any layer '''
    def __init__(self, dropout_p=0, enable_vbp=True):
        super(DropoutWrapper, self).__init__()
        self.dropout_p = dropout_prob # dropout probability (default = 0.4)
        self.enable_variational_dropout = enable_vbp # don't use fixed probability dropout

    def forward(self, x):
        if self.training == False or not self.dropout_p: # dropout is only applied while training
            return x

        if len(x.size()) == 3:
            # Bernoulli converts [0,1] -> {0,1} it discretizes the probability 
            mask = torch.bernoulli((1-self.dropout_p) * Variable(1.0 / (1-self.dropout_p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1)), requires_grad=False) # Use bernoulli distribution to approximate which elements to consider
            return mask.unsqueeze(1).expand_as(x) * x # set dimensions appropriately
        else:
            return torch.nn.functional.dropout(x, p=self.dropout_p, training=self.training)
