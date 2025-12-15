from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn as nn

from torch.nn import init
import math
import sys
sys.path.append("..")



class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvLSTMCell,self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + 2*hidden_size, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state_spatial, hidden_state_temporal):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state_spatial is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            
            prev_state_spatial = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )
                
        if hidden_state_temporal is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            hidden_state_temporal = Variable(torch.zeros(state_size)).cuda()
           
        prev_hidden_spatial, prev_cell_spatial = prev_state_spatial

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_hidden_spatial, hidden_state_temporal], 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell_spatial) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        state = [hidden,cell]

        return state

class ConvLSTMCellMask(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvLSTMCellMask,self).__init__()
        self.use_gpu = args.use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + 2*hidden_size + 1, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_mask, prev_state_spatial, hidden_state_temporal):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state_spatial is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )
                
        if hidden_state_temporal is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                hidden_state_temporal = Variable(torch.zeros(state_size)).cuda()
            else:
                hidden_state_temporal = Variable(torch.zeros(state_size))


        prev_hidden_spatial, prev_cell_spatial = prev_state_spatial

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_mask, prev_hidden_spatial, hidden_state_temporal], 1)
        del prev_hidden_spatial, hidden_state_temporal
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell_spatial) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)
        del cell_gate, out_gate, remember_gate, in_gate, gates, stacked_inputs

        state = [hidden,cell]

        return state


class RSIS(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self,):

        super(RSIS,self).__init__()
        self.hidden_size = 256
        self.kernel_size = 3
        padding = 0 if self.kernel_size == 1 else 1

        self.dropout = 0.1
        self.skip_mode = 'concat'

        # convlstms have decreasing dimension as width and height increase
        skip_dims_out = [self.hidden_size, int(self.hidden_size/2),
                         int(self.hidden_size/4),int(self.hidden_size/8)]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 4 is the number of deconv steps that we need to reach image size in the output
        for i in range(len(skip_dims_out)):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = skip_dims_out[i-1]
                if self.skip_mode == 'concat':
                    clstm_in_dim*=2

            args = None
            clstm_i = ConvLSTMCell(args, clstm_in_dim, skip_dims_out[i],self.kernel_size, padding = padding)
            self.clstm_list.append(clstm_i)

        self.conv_out = nn.Conv2d(skip_dims_out[-1], 1,self.kernel_size, padding = padding)

        # calculate the dimensionality of classification vector
        # side class activations are taken from the output of the convlstm
        # therefore we need to compute the sum of the dimensionality of outputs
        # from all convlstm layers
        fc_dim = 0
        for sk in skip_dims_out:
            fc_dim+=sk


   
    def forward(self, skip_feats, prev_state_spatial, prev_hidden_temporal):     
                  
        clstm_in = skip_feats[0]
        skip_feats = skip_feats[1:]
        hidden_list = []

        for i in range(len(skip_feats)+1):

            # hidden states will be initialized the first time forward is called
            if prev_state_spatial is None:
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in,None, None)
                else:
                    state = self.clstm_list[i](clstm_in,None, prev_hidden_temporal[i])
            else:
                # else we take the ones from the previous step for the forward pass
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in, prev_state_spatial[i], None)
                    
                else:
                    state = self.clstm_list[i](clstm_in, prev_state_spatial[i], prev_hidden_temporal[i])

            hidden_list.append(state)
            hidden = state[0]

            if self.dropout > 0:
                hidden = nn.Dropout2d(self.dropout)(hidden)

            # apply skip connection
            if i < len(skip_feats):

                skip_vec = skip_feats[i]
                upsample = nn.UpsamplingBilinear2d(size = (skip_vec.size()[-2],skip_vec.size()[-1]))
                hidden = upsample(hidden)
                # skip connection
                if self.skip_mode == 'concat':
                    clstm_in = torch.cat([hidden,skip_vec],1)
                elif self.skip_mode == 'sum':
                    clstm_in = hidden + skip_vec
                elif self.skip_mode == 'mul':
                    clstm_in = hidden*skip_vec
                elif self.skip_mode == 'none':
                    clstm_in = hidden
                else:
                    raise Exception('Skip connection mode not supported !')
            else:
                self.upsample = nn.UpsamplingBilinear2d(size = (hidden.size()[-2]*2,hidden.size()[-1]*2))
                hidden = self.upsample(hidden)
                clstm_in = hidden

        out_mask = self.conv_out(clstm_in)
        # classification branch

        return out_mask, hidden_list
        
class RSISMask(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self, args):

        super(RSISMask,self).__init__()
        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        padding = 0 if self.kernel_size == 1 else 1

        self.dropout = args.dropout
        self.skip_mode = args.skip_mode

        # convlstms have decreasing dimension as width and height increase
        skip_dims_out = [self.hidden_size, int(self.hidden_size/2),
                         int(self.hidden_size/4),int(self.hidden_size/8)]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 4 is the number of deconv steps that we need to reach image size in the output
        for i in range(len(skip_dims_out)):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = skip_dims_out[i-1]
                if self.skip_mode == 'concat':
                    clstm_in_dim*=2

            clstm_i = ConvLSTMCellMask(args, clstm_in_dim, skip_dims_out[i],self.kernel_size, padding = padding)
            self.clstm_list.append(clstm_i)
            del clstm_i

        self.conv_out = nn.Conv2d(skip_dims_out[-1], 1,self.kernel_size, padding = padding)

        # calculate the dimensionality of classification vector
        # side class activations are taken from the output of the convlstm
        # therefore we need to compute the sum of the dimensionality of outputs
        # from all convlstm layers
        fc_dim = 0
        for sk in skip_dims_out:
            fc_dim+=sk


   
    def forward(self, skip_feats, prev_mask, prev_state_spatial, prev_hidden_temporal):     
                  
        clstm_in = skip_feats[0]
        skip_feats = skip_feats[1:]
        hidden_list = []

        for i in range(len(skip_feats)+1):

            # hidden states will be initialized the first time forward is called
            if prev_state_spatial is None:
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in, prev_mask[i], None, None)
                else:
                    state = self.clstm_list[i](clstm_in, prev_mask[i], None, prev_hidden_temporal[i])
            else:
                # else we take the ones from the previous step for the forward pass
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in, prev_mask[i], prev_state_spatial[i], None)
                    
                else:
                    state = self.clstm_list[i](clstm_in, prev_mask[i], prev_state_spatial[i], prev_hidden_temporal[i])
                    #print(prev_hidden_temporal[i].shape)
            hidden_list.append(state)
            hidden = state[0]

            if self.dropout > 0:
                hidden = nn.Dropout2d(self.dropout)(hidden)

            # apply skip connection
            if i < len(skip_feats):

                skip_vec = skip_feats[i]
                upsample = nn.UpsamplingBilinear2d(size = (skip_vec.size()[-2],skip_vec.size()[-1]))
                hidden = upsample(hidden)
                # skip connection
                if self.skip_mode == 'concat':
                    clstm_in = torch.cat([hidden,skip_vec],1)
                elif self.skip_mode == 'sum':
                    clstm_in = hidden + skip_vec
                elif self.skip_mode == 'mul':
                    clstm_in = hidden*skip_vec
                elif self.skip_mode == 'none':
                    clstm_in = hidden
                else:
                    raise Exception('Skip connection mode not supported !')
            else:
                self.upsample = nn.UpsamplingBilinear2d(size = (hidden.size()[-2]*2,hidden.size()[-1]*2))
                hidden = self.upsample(hidden)
                clstm_in = hidden
            del hidden

        out_mask = self.conv_out(clstm_in)
        
        del clstm_in, skip_feats

        return out_mask, hidden_list
