import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pdb import set_trace as st

def get_X_y(data, one_hot=None, increment=False):
    n_obs = 6
    n_objs = 9
    base_obs = n_objs * n_obs

    if increment:
        N, _, T, __ = data.shape
        X = data[:, :, :, :base_obs].reshape(-1, 2, n_objs, T * n_obs)
        y = data[:, 0, 0, base_obs:base_obs + 2].astype('int')
    else:
        N, T, _ = data.shape
        X = data[:, :, :base_obs].reshape(-1, n_objs, T * n_obs)
        y = data[:, 0, base_obs:base_obs + 2].astype('int')

    if one_hot:
        y_oh = np.zeros((N, one_hot))
        y_oh[np.arange(N), y[:, 0].astype('int')] = 1
        y_oh[np.arange(N), y[:, 1].astype('int')] = 1
        y = y_oh

    return X, y

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

def kcl_data_to_increment_loaders(data, test_size, hist_length, num_objs, batch_size):
    '''
    Separate into "events" of x0 and x1.
    '''
    _, T, M = data.shape

    # split into train test by rollouts
    data_tr, data_t = train_test_split(data, test_size=test_size)

    # split into chunks of hist_length
    T_use = T // (2 * hist_length) * (2 * hist_length)
    data_tr_chunk = data_tr[:, :T_use, :].reshape(-1, 2, hist_length, M)
    data_t_chunk = data_t[:, :T_use, :].reshape(-1, 2, hist_length, M)

    # get X and y
    label_dim = num_objs
    X_tr, y_tr = get_X_y(data_tr_chunk, one_hot=label_dim, increment=True)
    X_t, y_t = get_X_y(data_t_chunk, one_hot=label_dim, increment=True)

    ds_tr = TensorDataset(from_numpy(X_tr), from_numpy(y_tr))
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    ds_t = TensorDataset(from_numpy(X_t), from_numpy(y_t))
    loader_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True)

    return loader_tr, loader_t

def kcl_data_to_loaders(data, test_size, hist_length, num_objs, batch_size):
    _, T, M = data.shape

    # split into train test by rollouts
    data_tr, data_t = train_test_split(data, test_size=test_size)

    # split into chunks of hist_length
    T_use = T // hist_length * hist_length
    data_tr_chunk = data_tr[:, :T_use, :].reshape(-1, hist_length, M)
    data_t_chunk = data_t[:, :T_use, :].reshape(-1, hist_length, M)

    # get X and y
    label_dim = num_objs
    X_tr, y_tr = get_X_y(data_tr_chunk, one_hot=label_dim)
    X_t, y_t = get_X_y(data_t_chunk, one_hot=label_dim)
    
    ds_tr = TensorDataset(from_numpy(X_tr), from_numpy(y_tr))
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    ds_t = TensorDataset(from_numpy(X_t), from_numpy(y_t))
    loader_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True)

    return loader_tr, loader_t


def fc_data_to_loaders(x_data, u_data, xnext_data, test_size, batch_size):

    X = np.hstack([x_data, u_data])
    Y = xnext_data
    # split into train test by rollouts
    X_tr, X_t, Y_tr, Y_t = train_test_split(X, Y, test_size=test_size)

   
    
    ds_tr = TensorDataset(from_numpy(X_tr),  from_numpy(Y_tr))
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    ds_t = TensorDataset(from_numpy(X_t), from_numpy(Y_t))
    loader_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True)

    return loader_tr, loader_t, X_tr, X_t, Y_tr, Y_t

def trajectories_to_data_full(root_dir):
    demo_folders = sorted(os.listdir(root_dir))
    x_data = []
    xnext_data = []
    u_data = []
    for folder in demo_folders:
        relative_end_effector_states = np.load(join(root_dir, folder, 'sensor', 'relative_end_effector_states.npy'))
        action_states = np.load(join(root_dir, folder, 'sensor', 'action_states.npy'))
        cube_states = np.load(join(root_dir, folder, 'sensor', 'cube_states.npy'))
        x_data.append(np.hstack((cube_states[:-1], relative_end_effector_states[:-1])))
        xnext_data.append(np.hstack((cube_states[1:], relative_end_effector_states[1:])))
        u_data.append(action_states)
    x_data = np.array(x_data)
    x_data = np.reshape(x_data, (x_data.shape[0] * x_data.shape[1], -1))
    xnext_data = np.array(xnext_data)
    xnext_data = np.reshape(xnext_data, (xnext_data.shape[0] * xnext_data.shape[1], -1))    
    u_data = np.array(u_data)
    u_data = np.reshape(u_data, (u_data.shape[0] * u_data.shape[1], -1))
    return x_data, xnext_data, u_data


def trajectories_to_data_XYZ(root_dir):
    demo_folders = sorted(os.listdir(root_dir))
    x_data = []
    xnext_data = []
    u_data = []
    for folder in demo_folders:
        relative_end_effector_states = np.load(join(root_dir, folder, 'sensor', 'relative_end_effector_states.npy'))
        action_states = np.load(join(root_dir, folder, 'sensor', 'action_states.npy'))
        cube_states = np.load(join(root_dir, folder, 'sensor', 'cube_states.npy'))
        x_data.append(np.hstack((cube_states[:-1, :3], relative_end_effector_states[:-1, :])))
        xnext_data.append(np.hstack((cube_states[1:, :3], relative_end_effector_states[1:, :])))
        xnext_data[-1] = xnext_data[-1]- x_data[-1]
        u_data.append(action_states[:, :2])
    x_data = np.array(x_data)
    x_data = np.reshape(x_data, (x_data.shape[0] * x_data.shape[1], -1))
    xnext_data = np.array(xnext_data)
    xnext_data = np.reshape(xnext_data, (xnext_data.shape[0] * xnext_data.shape[1], -1))    
    u_data = np.array(u_data)
    u_data = np.reshape(u_data, (u_data.shape[0] * u_data.shape[1], -1))
    return x_data, xnext_data, u_data


def visualize_3d_data():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x =[1,2,3,4,5,6,7,8,9,10]
    y =[5,6,2,3,13,4,1,2,4,8]
    z =[2,3,3,3,5,7,9,11,9,10]



    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

def visualize_all_data(root_dir):
    demo_folders = sorted(os.listdir(root_dir))
    x_data = []
    xnext_data = []
    u_data = []
    f, axarr = plt.subplots(2, 2)
    n_colors = 100
    cmap = plt.cm.get_cmap('hsv', n_colors)

    for i, folder in enumerate(demo_folders):
        relative_end_effector_states = np.load(join(root_dir, folder, 'sensor', 'relative_end_effector_states.npy'))
        action_states = np.load(join(root_dir, folder, 'sensor', 'action_states.npy'))
        cube_states = np.load(join(root_dir, folder, 'sensor', 'cube_states.npy'))
        x_data = np.hstack((cube_states[:-1, :3], relative_end_effector_states[:-1, :]))
        xnext_data = np.hstack((cube_states[1:, :3], relative_end_effector_states[1:, :]))
        u_data = action_states[:, :3]

        idx = np.random.choice(n_colors)
        axarr[0, 0].scatter(u_data[:, 0], u_data[:, 1], s=3, c=cmap(idx))
        axarr[0, 0].set_title('action data (dx, dy)')
        # axarr[0, 0].set_aspect('equal')
        axarr[0, 1].scatter(x_data[:, 0], x_data[:, 1], s=3, c=cmap(idx))
        axarr[0, 1].set_title('cube (x, y)')
        # axarr[0, 1].set_aspect('equal')
        axarr[1, 1].scatter(x_data[:, 3], x_data[:, 4], s=3, c=cmap(idx))
        axarr[1, 1].set_title('cube - ee (x, y)')
        # axarr[1, 1].set_aspect('equal')

    return f, axarr
"""
GPU wrappers from 
https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
"""

_use_gpu = False
device = None

def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")

def gpu_enabled():
    return _use_gpu

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def get_numpy(tensor):
    # not sure if I should do detach or not here
    return tensor.to('cpu').detach().numpy()

def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs).to(device)

def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs).to(device)

def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to(device)

def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).to(device)

def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)