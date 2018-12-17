import os, argparse, logging
import sys
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange, tqdm
from ipdb import set_trace as st
from models.fc_dynamics import ForwardDynamicsModel
from util.utils import weight_init, set_gpu_mode, zeros, get_numpy, fc_data_to_loaders, trajectories_to_data_XYZ, visualize_all_data
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from scipy import optimize
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
import pickle

_LOSS = nn.MSELoss

def compute_acc(labels_pred, y):
    N = len(labels_pred)
    corrects = labels_pred * y
    acc = torch.sum(corrects) / 2 / N
    return get_numpy(acc)

def labels_from_preds(preds):
    prob = F.softmax(preds, dim=1)
    _, indices = torch.topk(prob, 2)
    labels_pred = zeros(preds.shape, requires_grad=False)

    N = len(preds)
    labels_pred[np.arange(N), indices[:, 0]] = 1
    labels_pred[np.arange(N), indices[:, 1]] = 1

    return labels_pred

def forward_results(x, y, model):
    preds = model(x)
    labels_pred = labels_from_preds(preds)

    criterion = _LOSS()
    loss = get_numpy(criterion(preds, y))
    acc = compute_acc(labels_pred, y)

    return loss, acc, preds, labels_pred

def get_input_optimizer(action):
    optimizer = optim.Adam([action.requires_grad_()], lr=0.01)
    return optimizer

def train(model, loader_tr, loader_t, lr=1e-4, epochs=100):
    logs = {
        'loss': {
            'tr': [],
            't': []
        },
        'acc': {
            'tr': [],
            't': []
        }
    }
    criterion = _LOSS()
    opt = optim.Adam(model.parameters(), lr=lr)
    t_epochs = trange(epochs, desc='{}/{}'.format(0, epochs))
    num_batches_tr = len(loader_tr)
    num_batches_t = len(loader_t)
    for e in t_epochs:
        # Train
        loss_tr = 0
        acc_tr = 0
        t_batches = tqdm(loader_tr, leave=False, desc='Train')
        for xb, yb in t_batches:
            opt.zero_grad()
            pred = model(xb)

            loss = criterion(pred, yb)
            labels_pred = labels_from_preds(pred)
            acc = compute_acc(labels_pred, yb)
            loss_tr += loss
            acc_tr += acc

            loss.backward()
            opt.step()

            t_batches.set_description('Train: {:.2f}, {:.2f}'.format(loss, acc))
            t_batches.update()

            if e == 10:
                action_plan = torch.rand(3).cuda() * 0.01
                cur_state = xb[-1][:-3]
                action_opt = get_input_optimizer(action_plan)
                # scheduler = ReduceLROnPlateau(optimizer, 'min')
                def closure():
                    action_opt.zero_grad()
                    pred1 = model(torch.cat((cur_state, action_plan), dim =0))
                    loss1 = criterion(pred1, yb[-1])
                    loss1.backward()
                    return loss1
                print(xb[-1][-3:])
                for j in range(1000):
                    action_opt.zero_grad()
                    pred1 = model(torch.cat((cur_state, action_plan), dim =0))
                    loss1 = criterion(pred1, yb[-1])
                    loss1.backward()                    
                    print(action_plan.data.cpu())
                    action_opt.step()
                    # scheduler.step(loss1)
                pred1 = model(torch.cat((cur_state, action_plan), dim =0))
                loss1 = criterion(pred1, yb[-1])

                st()

        loss_tr /= num_batches_tr
        acc_tr /= num_batches_tr

        # Eval on test
        loss_t = 0
        acc_t = 0
        for xb, yb in tqdm(loader_t, leave=False, desc='Eval'):
            loss, acc, _, _ = forward_results(xb, yb, model)
            loss_t += loss
            acc_t += acc
        loss_t /= num_batches_t
        acc_t /= num_batches_t
        
        t_epochs.set_description('{}/{} | Tr {:.2f}, {:.2f}. T {:.2f}, {:.2f}'.format(e, epochs, loss_tr, acc_tr, loss_t, acc_t))
        t_epochs.update()
        print('epoch: ', e)
        print('train_loss: ', loss_tr)
        print('test_loss: ', loss_t)
        logs['loss']['tr'].append(loss_tr)
        logs['acc']['tr'].append(acc_tr)
        logs['loss']['t'].append(loss_t)
        logs['acc']['t'].append(acc_t)
        print('-'*10)

    return logs

def train_linear_model(X_tr, X_t, Y_tr, Y_t):
    # reg = LinearRegression().fit(X_tr, Y_tr )
    # print(reg.score(X_tr, Y_tr))
    # print(reg.score(X_t, Y_t))
    # print('-')
    clf = KernelRidge(alpha=0.01)
    clf = GaussianProcessRegressor(alpha=0.000001)
    clf.fit(np.vstack((X_tr, X_t)), np.vstack((Y_tr, Y_t)))
    print(clf.score(X_tr, Y_tr))
    print(clf.score(X_t, Y_t))   

    # grad = get_gradient(clf, X_tr[0], Y_tr[0])
    for _ in range(5):
        idx = np.random.randint(0, len(X_tr))
        u = X_tr[idx][-6:]
        x = X_tr[idx][:-6]
        xnext = Y_tr[idx]
        u_opt = get_optimal_action(u, x, xnext, clf)
        print('converged')
        # st()
    pickle.dump(clf, open('/home/msieb/projects/bullet-demonstrations/GP/model.pkl', 'wb'))
    return clf



def get_gradient(clf, x, y):
    def func(x, y):
        return mean_squared_error(clf.predict(x[None])[0], y)
    eps = 1e-6
    return optimize.approx_fprime(x, func, eps, y)

def fun(u, clf, x, xnext):
    xu = np.concatenate((x, u))
    return clf.predict(xu[None])[0] - xnext

def get_optimal_action(u, x, xnext, clf):
    res = least_squares(fun, u, method='lm', f_scale=0.1, args=(clf, x, xnext))['x']
    return res

def normalize_data(x_data, xnext_data, u_data):
    statistics = dict()
    statistics['reg'] = 1e-5
    statistics['x_mean'] = np.mean(x_data,axis=0)
    statistics['x_std'] = np.std(x_data,axis=0)
    statistics['u_mean'] = np.mean(u_data,axis=0)
    statistics['u_std'] = np.std(u_data,axis=0)
    statistics['xnext_mean'] = np.mean(xnext_data,axis=0)
    statistics['xnext_std'] = np.std(xnext_data,axis=0)
    x_data = (x_data - statistics['x_mean']) / (statistics['x_std'] + statistics['reg'])
    u_data = (u_data - statistics['u_mean']) / (statistics['u_std'] + statistics['reg'])
    xnext_data = (xnext_data - statistics['xnext_mean']) / (statistics['xnext_std'] + statistics['reg'])
    return x_data, xnext_data, u_data, statistics

if __name__ == '__main__':
    set_gpu_mode(True)
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    root_dir = '/home/msieb/projects/bullet-demonstrations/experiments/cube_push/test'
    # parser.add_argument('--x_data', '-x', type=str, default=join(data_dir, 'relative_end_effector_states.npy'))
    # parser.add_argument('--u_data', '-u', type=str, default=join(data_dir, 'action_states.npy'))
    # parser.add_argument('--cube_data', '-c', type=str, default=join(data_dir, 'cube_states.npy'))
    parser.add_argument('--out_dir', '-o', type=str, default='output/rn_rigid')
    parser.add_argument('--test_size', '-t', type=float, default=0.05)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--learning_rate', '-r', type=float, default=1e-4)
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    args = parser.parse_args()

    logging.info('Loading {}'.format(root_dir))
    logging.info('Processing Data')
    x_data, xnext_data, u_data = trajectories_to_data_XYZ(root_dir)

    x_data, xnext_data, u_data, statistics = normalize_data(x_data, xnext_data, u_data)
    pickle.dump(statistics, open('/home/msieb/projects/bullet-demonstrations/GP/statistics.pkl', 'wb'))



    loader_tr, loader_t, X_tr, X_t, Y_tr, Y_t = fc_data_to_loaders(x_data, u_data, xnext_data, args.test_size, args.batch_size)
    train_linear_model(X_tr, X_t, Y_tr, Y_t)  
    # fig, axarr = visualize_all_data(root_dir)
    # plt.show()
    sys.exit()
    logging.info('Training.')
    model = ForwardDynamicsModel(loader_tr.dataset.tensors[0].shape[-1], 
                            loader_tr.dataset.tensors[1].shape[-1]).cuda()
    model.apply(weight_init)

    logs = train(model, loader_tr, loader_t, lr=args.learning_rate, epochs=args.epochs)
    # TODO save stuff

    import IPython
    IPython.embed()
    exit()

