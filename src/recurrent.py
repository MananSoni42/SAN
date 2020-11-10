'''
Created October, 2017
Author: xiaodl@microsoft.com
'''
import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence as padd
from torch.nn.utils.rnn import pack_padded_sequence as pack
from .my_optim import weight_norm as WN

class OneLayerBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, prefix='stack_rnn', opt={}, dropout=None):
        super(OneLayerBRNN, self).__init__()
        self.opt = opt
        self.prefix = prefix
        self.cell_type = self.opt.get('{}_cell'.format(self.prefix), 'lstm').upper()
        self.emb_dim = self.opt.get('{}_embd_dim'.format(self.prefix), 0)
        self.maxout_on = self.opt.get('{}_maxout_on'.format(self.prefix), False)
        self.weight_norm_on = self.opt.get('{}_weight_norm_on'.format(self.prefix), False)
        self.dropout = dropout
        self.output_size = hidden_size if self.maxout_on else hidden_size * 2
        self.hidden_size = hidden_size
        self.rnn = getattr(nn, self.cell_type)(input_size, hidden_size, num_layers=1, bidirectional=True)

    def forward(self, x, x_mask):
        x = x.transpose(0, 1)
        size = list(x.size())
        #x = self.dropout(x)
        rnn_output, h = self.rnn(x)
        if self.maxout_on:
            rnn_output = rnn_output.view(size[0], size[1], self.hidden_size, 2).max(-1)[0]
        # Transpose back
        hiddens = rnn_output.transpose(0, 1)
        return hiddens

class BRNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, prefix='rnn', opt={}, dropout=None):
        super(BRNNEncoder, self).__init__()
        self.opt = opt
        self.dropout = dropout
        self.cell_type = opt.get('{}_cell'.format(self.prefix), 'gru').upper()
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(self.prefix), False)
        self.top_layer_only = opt.get('{}_top_layer_only'.format(self.prefix), False)
        self.num_layers = opt.get('{}_num_layers'.format(self.prefix), 1)
        self.rnn = getattr(nn, self.cell_type, default=nn.GRU)(input_size, hidden_size, self.num_layers, bidirectional=True)
        if self.weight_norm_on:
            self.rnn = WN(self.rnn)
        if self.top_layer_only:
            self.output_size = hidden_size * 2
        else:
            self.output_size = self.num_layers * hidden_size * 2

    def forward(self, x, x_mask):
        x = self.dropout(x)
        _, h = self.rnn(x.transpose(0, 1).contiguous())
        if self.cell_type == 'lstm':
            h = h[0]
        shape = h.size()
        h = h.view(self.num_layers, 2, shape[1], shape[3]).transpose(1,2).contiguous()
        h = h.view(self.num_layers, shape[1], 2 * shape[3])
        if self.top_layer_only:
            return h[-1]
        else:
            return h.transose(0, 1).contiguous().view(x.size(0), -1)


#------------------------------
# Contextual embedding
# TODO: remove packing to speed up
# Credit from: https://github.com/salesforce/cove
#------------------------------
class ContextualEmbedV2(nn.Module):
    #constructor
    def __init__(self,model_path,padding_idx=0):
        super(ContextualEmbedV2,self).__init__() # can use super().__init__
        
        state_dict=torch.load(model_path) #loading pre-trained model
        self.rnn=nn.LSTM(300,300,num_layers=1,bidirectional=True) #initialization of BiLSTM for Questions
        #Number of expected features in input x
        #Number of features in hidden state h
        #Number of recurrent layers.
        self.rnn2=nn.LSTM(600,300,num_layers=1,bidirectional=True) # for Passages
        
        #isinstance(obj,class) return boolean checking whether objects belong to that class.
        
        #state_dict is a dict obj that maps each layer to its parameter tensor
        #dict() to copy dict
        
        state_dict1=dict([(name,param.data) if isinstance (param,Parameter) else (name,param)
                          for name,param in state_dict.items() if '0' in name])

        #classifying weights and traininable parameters for question and passage respectively
    

        state_dict2=dict([(name.replace('1', '0'), param.data) if isinstance(param, Parameter) else (name.replace('1', '0'), param)
                        for name, param in state_dict.items() if '1' in name])

        self.rnn1.load_state_dict(state_dict1)
        self.rnn2.load_state_dict(state_dict2)

        for q in self.parameters():
            q.requires_grad=False

        self.output_size=600
        #2 times in original code why?


    def setup_eval_embed(self,eval_embed,padding_idx=0)
        pass

    def forward(self,x,x_mask):
        length=x_mask.data.eq(0).long().sum(1)  # tells number of 0 in a row
        maximum_length=x_mask.size(1)

        lens,indices=torch.sort(length,0,True) #returns indices of elements when elements were not sorted
        #sort in descending order
        
        # _ as engulf operator
        output1,_=self.rnn1(pack(x[indices],lens.tolist(),batch_first=True)
        output2,_=self.rnn2(output1)

        output1=padd(output1,batch_first=True)[0]
        output2=padd(output2,batch_first=True)[0]

        #pack_padded_sequence() -> packs the padded sequence(ignores 0), lens-> the order of packing

        #pad_packed_sequence()-> INVERSE OF ABOVE OPERATION(Adds 0 padding) 




        _,_indices=torch.sort(indices,0)  #sorting indices of 
        output1=output1[_indices]
        output2=output2[_indices]

        return output1,output2


class ContextualEmbed(nn.Module):
    def __init__(self, path, vocab_size, emb_dim=300, embedding=None, padding_idx=0):
        super(ContextualEmbed, self).__init__()  # can use super().__init__

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        #using inbuilt embedding of torch(may be word2vec)
        #size of embedding, size of each embedding vector
        #padding_idx-> pads the output with the embedding vector at padding_idx ,
        #whenever it encounters the index.

        if embedding is not None:
            self.embedding.weight.data = embedding

        state_dict = torch.load(path)       #same as previous contextual layer

        self.rnn1 = nn.LSTM(300, 300, num_layers=1, bidirectional=True)
        self.rnn2 = nn.LSTM(600, 300, num_layers=1, bidirectional=True)

        state_dict1 = dict([(name, param.data) if isinstance(param, Parameter) else (name, param)
                        for name, param in state_dict.items() if '0' in name])
        state_dict2 = dict([(name.replace('1', '0'), param.data) if isinstance(param, Parameter) else (name.replace('1', '0'), param)
                        for name, param in state_dict.items() if '1' in name])

        self.rnn1.load_state_dict(state_dict1)
        self.rnn2.load_state_dict(state_dict2)
        
        for p in self.parameters(): p.requires_grad = False

        self.output_size = 600


    def setup_eval_embed(self, eval_embed, padding_idx=0):
        
        self.eval_embed = nn.Embedding(eval_embed.size(0), eval_embed.size(1), padding_idx = padding_idx)
        self.eval_embed.weight.data = eval_embed

        for p in self.eval_embed.parameters():
            p.requires_grad = False


    def forward(self, x_idx, x_mask):
        
        emb = self.embedding if self.training else self.eval_embed
        x_hiddens = emb(x_idx)

        #Same as previously defined layer(below onwards)
        lengths = x_mask.data.eq(0).long().sum(1)
        max_len = x_mask.size(1)
        
        lens, indices = torch.sort(lengths, 0, True)
        
        output1, _ = self.rnn1(pack(x_hiddens[indices], lens.tolist(), batch_first=True))
        output2, _ = self.rnn2(output1)

        output1 = unpack(output1, batch_first=True, total_length=max_len)[0]
        output2 = unpack(output2, batch_first=True, total_length=max_len)[0]

        _, _indices = torch.sort(indices, 0)

        output1 = output1[_indices]
        output2 = output2[_indices]

        return output1, output2
