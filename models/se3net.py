import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d
from torchvision import models
from torch.autograd import Function, Variable
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as f

from copy import deepcopy as copy
import math
import sys
from pdb import set_trace as st 

sys.path.append('..')
from util.rot_utils import axisAngleToRotationMatrix_batched, rotationMatrixToAxisAngle_batched
from util.network_utils import apply

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

class BatchNormDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BatchNormDeconv2d, self).__init__()
        self.deconv2d = UpsampleConvLayer(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.deconv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(FCN, self).__init__()
        self.deconv2d = UpsampleConvLayer(in_channels, out_channels, **kwargs)
        # self.conv2d = Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.deconv2d(x)
        # x = self.conv2d(x)
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


class PoseAndMaskEncoder(nn.Module):
    """Pose Nets
    """
    def __init__(self, k):
        #Implementation a mix of SE3 Nets and SE3 Pose Nets
        super(PoseAndMaskEncoder, self).__init__()
        self.k = k # number of objects in scene
        # Encoder
        self.Conv1 = Conv2d(3, 8, bias=False, kernel_size=2, stride=1, padding=1)
        self.Pool1 = MaxPool2d(kernel_size=2)
        self.Conv2 = BatchNormConv2d(8, 16, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool2 = MaxPool2d(kernel_size=2)
        self.Conv3 = BatchNormConv2d(16, 32, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool3 = MaxPool2d(kernel_size=2)
        self.Conv4 = BatchNormConv2d(32, 64, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool4 = MaxPool2d(kernel_size=2)
        self.Conv5 = BatchNormConv2d(64, 128, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool5 = MaxPool2d(kernel_size=2)
        # Mask Decoder
        self.Deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        self.Deconv2 = BatchNormDeconv2d(64, 32, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        self.Deconv3 = BatchNormDeconv2d(32, 16, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        self.Deconv4 = BatchNormDeconv2d(16, 8, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        self.Deconv5 = FCN(8, self.k, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        # Pose Decoder
        self.Fc1 = Dense(128 * 7 * 7, 256)
        self.Fc2 = Dense(256, 128)
        self.Fc3 = Dense(128, 6 * self.k)  

    def encode_state(self, x):
                                               # x: 3 x 224 ** 2
        self.z1 = self.Pool1(self.Conv1(x))       # 8 x 112 ** 2
        self.z2 = self.Pool2(self.Conv2(self.z1)) # 16 x 56 ** 2
        self.z3 = self.Pool3(self.Conv3(self.z2)) # 32 x 28 ** 2
        self.z4 = self.Pool4(self.Conv4(self.z3)) # 64 x 14 ** 2
        self.z5 = self.Pool5(self.Conv5(self.z4)) # 128 x 7 ** 2
        z = self.z5
        return z

    def decode_mask(self, z):
                                               # z: 128 x 7 ** 2 
        self.m1 = self.Deconv1(z)                 # 64 x 14 ** 2
        self.m2 = self.Deconv2(self.m1 + self.z4) # 32 x 28 ** 2
        self.m3 = self.Deconv3(self.m2 + self.z3) # 16 x 56 ** 2
        self.m4 = self.Deconv4(self.m3 + self.z2) # 8 x 112 ** 2
        self.m5 = self.Deconv5(self.m4 + self.z1) # k x 224 ** 2
        k_sum = torch.sum(self.m5, dim=1)
        m = self.m5 / k_sum # sum of all mask weights per pixel equals 1
        return m

    def decode_pose(self, z):
        self.s1 = self.Fc1(z.view(-1, 128 * 7 * 7))
        self.s2 = self.Fc2(self.s1)
        self.s3 = self.Fc3(self.s2)
        p = self.s3
        return p

    def forward(self, x):
        enc_state = self.encode_state(x)
        mask = self.decode_mask(enc_state)
        poses = self.decode_pose(enc_state)
        return mask, poses

class PoseTransitionNetwork(nn.Module):
    def __init__(self, k, action_dim):
        super(PoseTransitionNetwork, self).__init__()
        self.k = k
        self.action_dim = action_dim

        # Pose change predictor
        self.Fc1_a = Dense(6 * self.k, 128)
        self.Fc2_a = Dense(128, 256)
        self.Fc1_b = Dense(self.action_dim, 128)
        self.Fc2_b = Dense(128, 256)
        self.Fc1_ab = Dense(512, 128)
        self.Fc2_ab = Dense(128, 64)
        self.Fc3_ab = Dense(64, 6 * self.k)

    def predict_pose_change(self, p, u):
        p_enc = self.Fc2_a(self.Fc1_a(p))
        u_enc = self.Fc2_b(self.Fc1_b(u))
        l = torch.cat((p_enc, u_enc), dim=-1)
        delta_p = self.Fc3_ab(self.Fc2_ab(self.Fc1_ab(l)))
        return delta_p
    
    def forward(self, poses, action):
        delta_poses = self.predict_pose_change(poses, action)
        return delta_poses

class TransformNetwork(nn.Module):
    def __init__(self, k, action_dim, gamma=1.0, sigma=0.5, training=True):
        super(TransformNetwork, self).__init__()
        self.k = k
        self.action_dim = action_dim
        self.pose_transition_network = PoseTransitionNetwork(k, action_dim)
        self.gamma = gamma
        self.sigma = sigma
        self.training = training

    def add_delta_poses(self, poses, delta_poses):
        """
        add delta pose to all poses k


        delta pose is given as a 6d vector where
        the latter 3 values=v are representing the axis,
        and |a| = theta. the first 3 represent the xyz translation
        
        requires seperate poses as input, so use apply function to use over all poses k
        """

        poses_unflattened = poses.view(self.k, 6)
        delta_poses_unflattened = poses.view(self.k, 6)

        # new_poses_unflattened = apply(self._add_delta_pose_per_object, (poses_unflattened, delta_poses_unflattened))
        # new_poses = new_poses_unflattened.view(-1)

        # get axang of current poses (poses is of dimension  6 * k, where k is the number of obj in scene)
        T = poses_unflattened[:, :3]
        axang = poses_unflattened[:, 3:]
        v = axang[:, :3] 
        R = axisAngleToRotationMatrix_batched(v)

        # get axang of delta poses
        delta_T = delta_poses_unflattened[:, :3]
        delta_axang = delta_poses_unflattened[:, 3:]
        delta_v = axang[:, :3] 
        delta_R = axisAngleToRotationMatrix_batched(v)        
    
        # obtain transformed poses
        axang_new = rotationMatrixToAxisAngle_batched(torch.matmul(delta_R, R)) # check if correct if R is batched
        pose_new = torch.stack((T + delta_T, axang_new), dim=1)
        pose_new = pose_new.view(-1).unsqueeze(0)
        return pose_new # (1, K * 6)

    def sharpen_mask_weights(self, mask):
        """Apply weight sharpening
        
        Parameters
        ----------
        mask : array K , H , W
            softmax over object association for each pixel

        Returns
        -------
        array K , H * W 
            sharpened weights
        """
        W = mask.shape[-1] 
        H = mask.shape[-2]
        mask = mask.view(-1, self.k, H * W)
        eps = torch.normal(mean=torch.zeros(self.k, H * W), std=torch.ones(self.k, H * W) * self.sigma ** 2)
        mask = (mask + eps.unsqueeze(0)) ** self.gamma
        k_sum = torch.sum(mask, dim=1)
        mask = mask / k_sum # sum of all mask weights per pixel equals 1    
        mask = mask.view(-1, self.k, H, W)
        if not self.training:
            ind = torch.max(mask, 1)[1]
            masmaskk_s = torch.zeros(mask.shape)
            mask[torch.arange(len(ind)), ind] = 1.0

        return mask

    def transform_point_cloud(self, x, mask, delta_poses):
        """Transform the given point cloud via applying SE3 and masking
        
        Parameters
        ----------
        x : array 3 , H , W
            point cloud of scene, flattened
        mask : array K , H , W
            object mask (soft)
        delta_poses : K , 6
            object poses as 6D vector of XYZ and axang representation
        
        Returns
        -------
        array 3 , H , W
            transformed point cloud of scene, unflattened
        """
        H = x.shape[-2]
        W = x.shape[-1]

        delta_poses_unflattened = delta_poses.view(self.k, 6)
        x = x.view(3, -1) # flatten image (H*W)
        # get axang of current poses (poses is of dimension  6 * k, where k is the number of obj in scene)
        delta_T = delta_poses_unflattened[:, :3]
        delta_axang = delta_poses_unflattened[:, 3:]
        delta_v = delta_axang[:, :3] 
        delta_R = axisAngleToRotationMatrix_batched(delta_v)        
        
        m = self.sharpen_mask_weights(mask)
        m = mask.view(self.k, -1)
        m = m.unsqueeze(-1).expand(self.k, m.shape[1], 3).transpose(2, 1)
        # dim: K * 3 * (H*W)
        # check batch size!!! TODO
        soft_transformed = torch.mul(m, torch.matmul(delta_R, x) + delta_T.unsqueeze(-1).expand(self.k, 3, x.shape[-1]))
        
        # dim: 3 * (H*W)
        x_new = torch.sum(soft_transformed, dim=0)
        x_new = x_new.view(3, H, W)
        return x_new

    def forward(self, x, mask, poses, action):
        delta_poses = self.pose_transition_network(poses, action)
        poses_new = self.add_delta_poses(poses, delta_poses)
        x_new = self.transform_point_cloud(x, mask, delta_poses)
        return x_new, poses_new 

class SE3Net(nn.Module):
    def __init__(self, k, action_dim, gamma=1.0, sigma=0.5, training=True):
        super(SE3Net, self).__init__()
        self.k = k
        self.action_dim = action_dim
        self.gamma = gamma
        self.sigma = sigma
        self.training = training

        self.transformer = TransformNetwork(k, action_dim)
        self.encoder = PoseAndMaskEncoder(k)

    def forward(self, x, action):
        mask, poses = self.encoder(x)
        x_new, poses_new = self.transformer(x, mask, poses, action)
        return poses_new.unsqueeze(0), x_new.unsqueeze(0)
       
# def define_model(pretrained=True, action_dim=6):
#     return TCNModel(models.inception_v3(pretrained=pretrained), action_dim)

model = SE3Net(5,4)
x = torch.rand(1,3,224,224)
u = torch.rand(1,4)
poses_new, x_new = model(x,u)
st()


# Generate some data and build train script
# check out what happens with background mask
# learn model of block thats pushing (priority)