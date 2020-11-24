'''
Generic dropout wrapper
Can be wrapped around any torch layer
'''

import torch
from torch.autograd import Variable

class DropoutWrapper(torch.nn.Module):
    ''' Aply this wrapper to enable dropout on any layer '''
    def __init__(self, prob=0, enable_vbp=True): # Second parameter is only for backwards compatibility with other modules
        super(DropoutWrapper, self).__init__()
        self.prob = prob # dropout probability (default = 0.4)

    def forward(self, x):
        # dropout is only applied while training
        if self.training == False or not self.prob:
            return x

        if len(x.size()) == 3:
            # Bernoulli converts [0,1] -> { 0, 1 } it discretizes the probability
            mask = torch.bernoulli((1-self.prob) * Variable(1.0 / (1-self.prob) * (x.data.new(x.size(0), x.size(2)).zero_() + 1)), requires_grad=False) # Use bernoulli distribution to approximate which elements to consider
            return mask.unsqueeze(1).expand_as(x) * x # set dimensions appropriately
        else:
            return torch.nn.functional.dropout(x, p=self.prob, training=self.training)
