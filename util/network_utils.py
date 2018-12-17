import torch
import numpy as np
import math
from ipdb import set_trace
from rot_utils import rotationMatrixToEulerAngles, axisAngletoRotationMatrix, \
              sincos2rotm


USE_CUDA = True

def geodesic_dist(R1, R2):
    mult = torch.matmul(torch.transpose(R1, dim0=1, dim1=2), R2)
    diagonals = torch.mul(mult, torch.eye(3).cuda())
    trace = torch.sum(torch.sum(diagonals, dim=1), dim=1)
    dist = torch.acos((trace - 1) / 2.0) # implements geodesic distance of two rotation matrix as loss
    return dist

def geodesic_dist_quat(q1, q2):
    dist = 2*torch.acos(torch.abs(torch.sum(q1*q2, dim=1)).clamp(-1.0+1e-7, 1.0-1e-7))
    return dist

def loss_quat_huber(model, anchor_frames, anchor_quats, lambd=0.01):
  """ 
  Calculates reparamerized euler angles as network output and puts
  loss on rotation matrix calculated from those, after normalizing the sin/cos values
  Assumes 6 dimensional rotation parameters -> sin/cos per angle,
  over all 6 values
  """

  features, a_pred = model(anchor_frames)
  assert a_pred.shape[-1] == 4
  # dist = geodesic_dist_quat(anchor_quats, 
  
  # #print("Correctly classified rotations: {}".format(np.sum(dist.data.cpu().numpy() < 0.2)))
  # #print("distances of batch: ", dist.data.cpu().numpy())
  loss = torch.nn.SmoothL1Loss()(a_pred, anchor_quats) #+ \
   #     lambd * torch.nn.SmoothL1Loss()(features_first_view_gt, features_second_view_gt.detach())
  return loss, a_pred

def loss_quat(model, minibatch, lambd=0.01):
  """ 
  Calculates reparamerized euler angles as network output and puts
  loss on rotation matrix calculated from those, after normalizing the sin/cos values
  Assumes 6 dimensional rotation parameters -> sin/cos per angle,
  over all 6 values
  """
  if USE_CUDA:
     anchor_frames = minibatch[0].cuda()
     anchor_quats = minibatch[1].cuda() # load as 3x3 rotation matrix
  features_second_view_gt, a_pred, features_first_view_gt = model(anchor_frames)
  assert a_pred.shape[-1] == 4
  dist = geodesic_dist_quat(anchor_quats, a_pred)
  #print("Correctly classified rotations: {}".format(np.sum(dist.data.cpu().numpy() < 0.2)))
  #print("distances of batch: ", dist.data.cpu().numpy())
  loss = dist.mean() #+ \
   #     lambd * torch.nn.SmoothL1Loss()(features_first_view_gt, features_second_view_gt.detach())
  return loss

def loss_quat_single(model, anchor_frames, anchor_quats, lambd=0.01):
  """ 
  Calculates reparamerized euler angles as network output and puts
  loss on rotation matrix calculated from those, after normalizing the sin/cos values
  Assumes 6 dimensional rotation parameters -> sin/cos per angle,
  over all 6 values
  """
  features, a_pred = model(anchor_frames)
  assert a_pred.shape[-1] == 4
  dist = geodesic_dist_quat(anchor_quats, a_pred)
  #print("Correctly classified rotations: {}".format(np.sum(dist.data.cpu().numpy() < 0.2)))
  #print("distances of batch: ", dist.data.cpu().numpy())
#3if np.isnan(dist.data.cpu().numpy()):
 #     set_trace()
  loss = dist.mean()
  if loss == 0:
      set_trace()
  if np.isnan(loss.data.cpu().numpy()):
    print('exploded gradients')
  return loss

def loss_rotation(model, anchor_frames, anchor_rots, lambd=0.01):
  """ 
  Calculates reparametrized euler angles as network output and puts
  loss on rotation matrix calculated from those, after normalizing the sin/cos values
  Assumes 6 dimensional rotation parameters -> sin/cos per angle,
  over all 6 values
  """
  features, a_pred = model(anchor_frames) 
  rots_pred = apply(sincos2rotm, a_pred)
  dist = geodesic_dist(anchor_rots, rots_pred)
  #print("Correctly classified rotations: {}".format(np.sum(dist.data.cpu().numpy() < 0.2)))

  #print("distances of batch: ", dist.data.cpu().numpy())
  loss = dist.mean() 
   #     lambd * torch.nn.SmoothL1Loss()(features_first_view_gt, features_second_view_gt.detach())
  return loss, a_pred


def loss_axisangle(model, minibatch, lambd=0.1):
  """ 
  Calculates axis/angle representation as network output and transforms to
  rotation matrix to put geodesic distance loss on it
  Assumes 4 output parameters: 3 for axis, 1 for angle
  """
  if USE_CUDA:
     anchor_frames = minibatch[0].cuda()
     #anchor_euler_reparam = minibatch[1].cuda() # load as 3x3 rotation matrix
     anchor_rots = minibatch[1].cuda() # load as 3x3 rotation matrix

  features_second_view_gt, a_pred, features_first_view_gt = model(anchor_frames) 
  assert a_pred.shape[-1] == 4
  a_pred[:, -1] = ((a_pred[:, -1] + 1.0) + math.pi) /2.0
  rots_pred = apply(axisAngletoRotationMatrix, a_pred)
  dist = geodesic_dist(anchor_rots, rots_pred)
  #print("distances of batch: ", dist.data.cpu().numpy())
  loss = dist.mean() + \
          lambd * torch.nn.SmoothL1Loss()(features_first_view_gt, features_second_view_gt.detach())
  return loss

def loss_euler_reparametrize(model, anchor_frames, anchor_rots, lambd=0.1):
  """ 
  Calculates reparamerized euler angles as network output and puts
  loss directly on those with Huber distance
  Assumes 6 dimensional rotation parameters -> sin/cos per angle,
  over all 6 values
  """

  euler = apply(rotationMatrixToEulerAngles, anchor_rots)
  # Order: X-Y-Z: roll pith yaw
  euler_reparam = euler_XYZ_to_reparam(euler)
  features, a_pred = model(anchor_frames) 
  assert a_pred.shape[-1] == 6
  loss = torch.nn.SmoothL1Loss()(a_pred, euler_reparam) #+ \
          #lambd * torch.nn.SmoothL1Loss()(features_first_view_gt, features_second_view_gt.detach())
  return loss, a_pred


def batch_size(epoch, max_size):
    exponent = epoch // 100
    return min(max(2 ** (exponent), 2), max_size)

def apply(func, M):
    tList = [func(m) for m in torch.unbind(M, dim=0) ]
    res = torch.stack(tList, dim=0)
    return res 

def euler_XYZ_to_reparam(euler):
  roll = euler[:, 0]
  pitch = euler[:, 1]
  yaw = euler[:, 2]
  euler_reparam = torch.stack(((torch.sin(roll),
                          torch.cos(roll),
                          torch.sin(pitch),
                          torch.cos(pitch),
                          torch.sin(yaw),
                          torch.cos(yaw),
                          )), dim=1) 
  return euler_reparam
