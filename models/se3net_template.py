import torch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Function, Variable
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from copy import deepcopy as copy
import math
from pdb import set_trace

VOCAB_SIZE = 2

class BatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BatchNormConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x, inplace=True)
        return x

class EmbeddingNet(nn.Module):
    def normalize(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output

class PosNet(EmbeddingNet):
    def __init__(self):
        super(PosNet, self).__init__()
        # Input 1
        self.Conv2d_1a = nn.Conv2d(3, 64, bias=False, kernel_size=10, stride=2)
        self.Conv2d_2a = BatchNormConv2d(64, 32, bias=False, kernel_size=3, stride=1)
        self.Conv2d_3a = BatchNormConv2d(32, 32, bias=False, kernel_size=3, stride=1)
        self.Conv7d_4a = BatchNormConv2d(32, 32, bias=False, kernel_size=2, stride=1)

        self.Dense1 = Dense(6 * 6 * 32, 32)
        self.alpha = 10

        # 128 x 128 x 3    def forward(self, input_batch):

        x = self.Conv2d_1a(input_batch)
        # 60 x 60 x 64
        x = self.Conv2d_2a(x)
        # 58 x 58 x 64
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 29 x 29 x 32
        x = self.Conv2d_3a(x)
        # 27 x 27 x 32
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 13 x 13 x 32
        x = self.Conv2d_4a(x)
        # 12 x 12 x 32
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size()[0], -1)
        # 6 x 6 x 32
        x = self.Dense1(x)
        # 32

        return self.normalize(x) * self.alpha
        
class TCNModel(EmbeddingNet):
    def __init__(self, inception, action_dim=6):  
        super(TCNModel, self).__init__()
        self.action_dim = action_dim
        self.state_dim = 1024
        self.transform_input = True
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.Conv2d_6a_3x3 = BatchNormConv2d(3, 100, kernel_size=3, stride=1)
        self.Conv2d_6b_3x3 = BatchNormConv2d(100, 200, kernel_size=3, stride=1)
        self.Conv2d_6c_3x3 = BatchNormConv2d(200, 100, kernel_size=3, stride=2)
        self.Conv2d_6d_3x3 = BatchNormConv2d(100, 20, kernel_size=3, stride=2)
        self.SpatialSoftmax = nn.Softmax2d()
        self.FullyConnected7a = Dense(36*36*20, self.state_dim)
        self.FullyConnectedSingle = Dense(self.state_dim, 512)
        # self.FullyConnectedConcat = Dense(2*self.state_dim, 128)
        # self.FullyConnectedPose1 = Dense(128, 256)
        self.FullyConnectedPose2 = Dense(512, 256)
        self.FullyConnectedPose3 = Dense(256, self.action_dim)
        self.tanh = torch.nn.Tanh()
        self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=math.pi)

        self.FullyConnectedAction1 = Dense(self.state_dim + self.action_dim, 256)
        self.FullyConnectedAction2 = Dense(256,512)
        self.FullyConnectedAction3 = Dense(512, 256)
        self.FullyConnectedAction4 = Dense(256, 32)

        self.alpha = 10.0

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:,  0] = x[:,  0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:,   1] = x[:,  1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:,  2] = x[:,  2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3

        # 33 x 33 x 100
        x = self.Conv2d_6a_3x3(x)
        # 31 x 31 x 20
        x = self.Conv2d_6b_3x3(x)
        # 31 x 31 x 20
        x = F.max_pool2d(x, kernel_size=3, stride=2) 
        x = self.Conv2d_6c_3x3(x)
        x = self.Conv2d_6d_3x3(x)
        # x = self.SpatialSoftmax(x)
        # 32
        x = self.FullyConnected7a(x.view(x.size()[0], -1))
        # Reshape to separate inputs
        # Predicts cos/sin values (tanh'ed)
        a_pred = self.FullyConnectedSingle(x)
        a_pred = self.FullyConnectedPose2(a_pred)
        a_pred = self.FullyConnectedPose3(a_pred)
        a_pred = self.normalize(a_pred)
        first_view_gt = x
        # Note that ground truth (gt) means the feature extracted from the intermediate FC, and pred means head output
       
        return first_view_gt, a_pred


def define_model(pretrained=True, action_dim=6):
    return TCNModel(models.inception_v3(pretrained=pretrained), action_dim)

