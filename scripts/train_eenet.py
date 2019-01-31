import os, argparse, logging
import sys
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

from tqdm import trange, tqdm
from ipdb import set_trace as st
from models.eenet import define_model
from util.utils import weight_init, set_gpu_mode, zeros, get_numpy
from util.eebuilder import EndEffectorPositionDataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "1, 2"

_LOSS = nn.NLLLoss

# plt.ion()

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

def imshow(img, pred, label):
    # img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(np.squeeze(img.astype(np.uint8)), (1, 2, 0)))
    # plt.scatter(pred[0], pred[1], s=10, marker='.', c='r')
    plt.scatter(label[1], label[0], s=50, marker='.', c='r')
    pred /= np.sum(pred)
    plt.imshow(np.squeeze(pred), cmap="YlGnBu", interpolation='bilinear', alpha=0.4)

def show_heatmap_of_samples(dataiter, model, use_cuda=True):
    n_display = 8
    for i in range(n_display):
        plt.subplot(2, 4, i+1)
        image, label = dataiter.next()
        if use_cuda:
            image = image.cuda()
            label = label.cuda()
        buf = np.where(label.cpu().numpy() ==1)[1:]
        label = (buf[0][0], buf[1][0])
        pred = model(image.cuda())
        imshow(image.cpu().detach().numpy(), pred.cpu().detach().numpy(), label)
    plt.show()


def train(model, loader_tr, loader_t, lr=1e-4, epochs=1000, use_cuda=True):
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
    dataiter = iter(loader_t)

    for e in t_epochs:
        # Train
        loss_tr = 0
        acc_tr = 0
        t_batches = tqdm(loader_tr, leave=False, desc='Train')
        import ipdb; ipdb.set_trace();
        # show heatmap of samples
        show_heatmap_of_samples(dataiter, model)
        
        for xb, yb in t_batches:


            if use_cuda:
                xb = xb.cuda()
                yb = yb.cuda()
            opt.zero_grad()
            pred = model(xb)

            loss = criterion(pred.view(pred.size()[0], -1), torch.max(yb.view(yb.size()[0], -1), 1)[1])
            # loss = criterion(pred, yb)

            # labels_pred = labels_from_preds(pred)
            # acc = compute_acc(labels_pred, yb)
            loss_tr += loss
            # acc_tr += acc

            loss.backward()
            opt.step()

            # t_batches.set_description('Train: {:.2f}, {:.2f}'.format(loss, acc))
            t_batches.update()


        loss_tr /= num_batches_tr
        acc_tr /= num_batches_tr

        # Eval on test
        loss_t = 0
        acc_t = 0

        
        # for xb, yb in tqdm(loader_t, leave=False, desc='Eval'):
        #     if use_cuda:
        #         xb = xb.cuda()
        #         yb = yb.cuda()
        #     pred = model(xb)
        #     loss = criterion(pred.view(pred.size()[0], -1), torch.max(yb.view(yb.size()[0], -1), 1)[1])
        #     loss_t += loss
            # acc_t += acc
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

def create_model(args, use_cuda=True):
    model = define_model(240, 240, use_cuda)
    # tcn = PosNet()
    if args.load_model:
        model_path = os.path.join(
            args.model_path,
        )
        # map_location allows us to load models trained on cuda to cpu.
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    if use_cuda:
        model = model.cuda()
    return model

if __name__ == '__main__':
    set_gpu_mode(True)
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    root_dir = '/home/msieb/projects/bullet-demonstrations/experiments/reach/data'
    # parser.add_argument('--x_data', '-x', type=str, default=join(data_dir, 'relative_end_effector_states.npy'))
    # parser.add_argument('--u_data', '-u', type=str, default=join(data_dir, 'action_states.npy'))
    # parser.add_argument('--cube_data', '-c', type=str, default=join(data_dir, 'cube_states.npy'))
    parser.add_argument('--out_dir', '-o', type=str, default='output/rn_rigid')
    parser.add_argument('--test_size', '-t', type=float, default=0.2)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--learning_rate', '-r', type=float, default=1e-4)
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    
    args = parser.parse_args()

    logging.info('Loading {}'.format(root_dir))
    logging.info('Processing Data')
    
    dataset = EndEffectorPositionDataset(root_dir=root_dir)
    n = len(dataset)
    n_test = int( n * .2 )  # number of test/val elements
    n_train = n - 2 * n_test
    dataset_tr, dataset_t, dataset_val = train_set, val_set, test_set = random_split(dataset, (n_train, n_test, n_test))
    loader_tr = DataLoader(dataset_tr, batch_size=2,
                        shuffle=True, num_workers=4)
    loader_t = DataLoader(dataset_t, batch_size=1, shuffle=True)                       

    logging.info('Training.')

    # TODO
    model = create_model(args)
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    logs = train(model, loader_tr, loader_t, lr=args.learning_rate, epochs=args.epochs)
    # TODO save stuff

    import IPython
    IPython.embed()
    exit()

