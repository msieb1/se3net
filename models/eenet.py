import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Function
import torchvision.models as models
from copy import deepcopy as copy


class SoftmaxLogProbability2D(torch.nn.Module):
    def __init__(self):
        super(SoftmaxLogProbability2D, self).__init__()

    def forward(self, x):
        orig_shape = x.data.shape
        seq_x = []
        for channel_ix in range(orig_shape[1]):
            softmax_ = F.softmax(x[:, channel_ix, :, :].contiguous()
                                 .view((orig_shape[0], orig_shape[2] * orig_shape[3])), dim=1)\
                .view((orig_shape[0], orig_shape[2], orig_shape[3]))
            seq_x.append(softmax_.log())
        x = torch.stack(seq_x, dim=1)
        return x

class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

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

class EENet(EmbeddingNet):
    def __init__(self, img_height, img_width, inception):  
        super(EENet, self).__init__()
        self.transform_input = True
        self.img_height = img_height
        self.img_width = img_width
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
        self.Conv2d_6a_3x3 = BatchNormConv2d(288, 100, kernel_size=3, stride=1)
        self.Conv2d_6b_3x3 = BatchNormConv2d(100, 20, kernel_size=3, stride=1)
        self.Deconv1 = UpsampleConvLayer(100, 10, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        self.Deconv2 = UpsampleConvLayer(10, 5, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        self.Deconv3 = UpsampleConvLayer(5, 1, kernel_size=3, stride=1, upsample=2) #, output_padding=1)

        # self.Deconv1 = nn.ConvTranspose2d(20, 1, kernel_size=3, stride=3) #, output_padding=1)
        # self.Deconv2 = nn.ConvTranspose2d(10, 1, kernel_size=3, stride=3) #, output_padding=1)

        self.Dropout = nn.Dropout(p=0.3)
        self.Dropout2d = nn.Dropout2d(p=0.3)
        self.Softmax2D = SoftmaxLogProbability2D()
        self.LogSoftmax = nn.LogSoftmax()
        self.SpatialSoftmax = nn.Softmax2d()
        self.Softmax1d = nn.LogSoftmax()
        self.FullyConnected7a = Dense(23 * 23 * 20, img_height * img_width)


    def forward(self, x):
        if self.transform_input:
            if x.shape[1] == 4:
                x = x[:, :-1].clone()
            else:
                x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        x = self.Dropout2d(x)

        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        x = self.Dropout2d(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        y = self.Mixed_5d(x)
        # 33 x 33 x 100
        x = self.Conv2d_6a_3x3(y)
        x = self.Dropout2d(x)
        # 31 x 31 x 20
        # x = self.Conv2d_6b_3x3(x)

        x = self.Deconv1(x)  
        x = self.Deconv2(x)
        x = self.Deconv3(x)
        x = F.interpolate(x, mode='nearest', size=(self.img_height, self.img_width))

        # x = nn.functional.interpolate(x, (self.img_height, self.img_width))
        # x = self.Softmax2D(x)
        x = x.view(x.size()[0], -1)
        x = self.Dropout(x)
        x = self.LogSoftmax(x)
        x = x.view(x.size()[0], self.img_height, self.img_width)
        # print(x.shape)
        
        # x = self.Deconv2(x)
        # 31 x 31 x 20
        # x = self.SpatialSoftmax(x)

        # width * height
        # x = self.FullyConnected7a(x.view(x.size()[0], -1))
        # Probabilistic softmax output
        # x = self.Softmax1d(x)
        # x = x.view(x.size()[0], self.img_height, self.img_width)
        # return x.squeeze_(1)
        return x

def define_model(img_height, img_width, pretrained=True):
    return EENet(img_height, img_width, models.inception_v3(pretrained=pretrained))