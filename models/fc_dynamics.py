import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from ipdb import set_trace as st
from util.utils import randn, ones, get_numpy


class ForwardDynamicsModel(nn.Module):

    def __init__(self, state_action_dim, state_dim, g_layers=[128, 64], f_layers=[32, 16], h=F.selu):
        super(ForwardDynamicsModel, self).__init__()
        self.h = h

        # forming G
        self.gs = nn.ModuleList()
        for i in range(len(g_layers)):
            in_size = state_action_dim if i == 0 else g_layers[i-1]
            out_size = g_layers[i]
            self.gs.append(nn.Linear(in_size, out_size))
        self.gs.append(nn.Linear(g_layers[-1], state_dim))

        self._hid_size = g_layers[-1]

        # forming F
        # self.fs = nn.ModuleList()
        # for i in range(len(f_layers)):
        #     in_size = g_layers[-1] if i == 0 else f_layers[i-1]
        #     out_size = f_layers[i]
        #     self.fs.append(nn.Linear(in_size, out_size))
        # self.fs.append(nn.Linear(f_layers[-1], label_dim))

    def forward(self, xu):
        '''

        '''
        # G pass
        out = self.h(self.gs[0](xu))
        for g in self.gs[1:]:
            out = self.h(g(out))

        # # F pass
        # for f in self.fs[:-1]:
        #     x = self.h(f(x))
        # out = self.fs[-1](x) # don't apply nonlinearity for output

        return out
