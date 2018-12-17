import os
import functools
import imageio
import numpy as np
import math
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset
from torch import Tensor
from torch.autograd import Variable
import logging
import nltk
import sys
import matplotlib.pyplot as plt
import pickle
import datetime
from ipdb import set_trace

def time_stamped_w_name(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

def time_stamped(fmt='%Y-%m-%d-%H-%M-%S{}'):
    return datetime.datetime.now().strftime(fmt).format('')

def distance(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)

def view_image(frame):
    # For debugging. Shows the image
    # Input shape (3, 299, 299) float32
    img = Image.fromarray(np.transpose(frame * 255, [1, 2, 0]).astype(np.uint8))
    img.show()

def write_to_csv(values, keys, filepath):
    if  not(os.path.isfile(filepath)):
        with open(filepath, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(keys)
            filewriter.writerow(values)
    else:
        with open(filepath, 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(values)


def ensure_folder(folder):
    path_fragments = os.path.split(folder)
    joined = '.'
    for fragment in path_fragments:
        joined = os.path.join(joined, fragment)
        if not os.path.exists(joined):
            os.mkdir(joined)

def resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def write_video(file_name, path, frames):
    imageio.mimwrite(os.path.join(path, file_name), frames, fps=60)

def read_video(filepath, frame_size):
    try:
        imageio_video = imageio.read(filepath)
    except:
        print("error loading video file, name: {}".format(filepath))
    snap_length = len(imageio_video) 
    frames = np.zeros((snap_length, 3, *frame_size))
    resized = map(lambda frame: resize_frame(frame, frame_size), imageio_video)
    for i, frame in enumerate(resized):
        frames[i, :, :, :] = frame
    return frames

def read_extracted_video(filepath, frame_size):
    try:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('.')[0]))
    except:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    all_images = [file for file in files if file.endswith('.jpg')]
    snap_length = len(all_images) 
    frames = []
    for i, filename in enumerate(all_images):
        frame = plt.imread(os.path.join(filepath, filename))
        frames.append(frame)
    return frames

def read_extracted_rcnn_results(filepath, frame_size):
    try:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('.')[0]))
    except:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    all_files = [file for file in files if file.endswith('.pkl')]
    snap_length = len(all_files) 
    all_results = []
    for i, filename in enumerate(all_files):
        with open(os.path.join(filepath, filename), 'rb') as fb:
            all_results.append(pickle.load(fb))    
    return all_results

def read_caption(filepath):
    try:
        with open(filepath, 'r') as fp:
            caption = fp.readline()
        return caption
    except:
        print("{} does not exist".format(filepath))
        return None

def ls_directories(path):
    return next(os.walk(path))[1]

# def ls(path):
#     # returns list of files in directory without hidden ones.
#     return sorted([p for p in os.listdir(path) if p[0] != '.' and (p[-4:] == '.mp4' or p[-4] == '.mov')], key=lambda x: int(x.split('_')[0] + x.split('.')[0].split('view')[1]))
#     # randomize retrieval for every epoch?

def ls(path):
    # returns list of files in directory without hidden ones.
    sort_seq =  sorted([p for p in os.listdir(path) if p[0] != '.' and (p[-4:] == '.mp4' or p[-4:] == '.mov')], key=lambda x: int(x.split('_')[0]))
    return sorted([p for p in os.listdir(path) if p[0] != '.' and (p[-4:] == '.mp4' or p[-4:] == '.mov')], key=lambda x:(x.split('.')[0]))

    # rand

def ls_unparsed_txt(path):
    return sorted([p for p in os.listdir(path) if p[0] != '.' and p[-5] != 'd' and p.endswith('.txt')], key=lambda x: int(x.split('.')[0]))


def ls_npy(path):
    # returns list of files in directory without hidden ones.
    return sorted([p for p in os.listdir(path) if p[0] != '.' and p[-4:] == '.npy'], key=lambda x: x.split('.')[0])
    # rand

def ls_txt(path):
    return sorted([p for p in os.listdir(path) if p[0] != '.' and p.endswith('.txt')], key=lambda x: x.split('.')[0])

def ls_view(path, view):
    # Only lists video files
    files = sorted([p for p in os.listdir(path) if p[0] != '.' and (p.endswith(str(view) + '.mp4')) and not p.endswith('cropped.mp4')], key=lambda x: int(x.split('_')[0]))
    if len(files) == 0:
        files = sorted([p for p in os.listdir(path) if p[0] != '.' and p.endswith('.mp4')], key=lambda x: int(x.split('_')[0]))
    return files

def ls_extracted(path):

     # returns list of folders in directory without hidden ones.
    return sorted([p for p in os.listdir(path) if (p[0] != '.' and p != 'debug') ], key=lambda x: int(x.split('_')[0]))


def crop_box(image, box, y_offset=0, x_offset=0):
    y1 = max(0, box[0]-y_offset)
    y2 = min(image.shape[0], box[2]+y_offset)
    x1 = max(0, box[1]-x_offset)
    x2 = min(image.shape[1], box[3]+x_offset)
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def crop_uniform_box(image, box, height_scaling=100, width_scaling=100):
    height = box[2] - box[0]
    width = box[3] - box[1]
    center_y = box[0] + 0.5 * height
    center_x = box[1] + 0.5 * width
    # Apply deltas


    height_e = height_scaling
    width_e = width_scaling

    y1_e = max(int(center_y - 0.5 * height_e), 0)
    x1_e = max(int(center_x - 0.5 * width_e), 0)
    y2_e = min(int(y1_e + height_e), image.shape[0] - 1)
    x2_e = min(int(x1_e + width_e), image.shape[1] - 1)
    cropped_image = image[y1_e:y2_e, x1_e:x2_e]
    return cropped_image

def get_box_center(box):
    y1 = box[0]
    y2 = box[2]
    x1 = box[1]
    x2 = box[3]
    return (x1 + x2)/2, (y1 + y2)/2

class Logger(object):
    def __init__(self, logfilename):
        logging.basicConfig(filename=logfilename, level=logging.DEBUG, filemode='a')

    def info(self, *arguments):
        print(*arguments)
        message = " ".join(map(repr, arguments))
        logging.info(message)



def collate_fn(data):
    """ eates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    frames, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    frames = torch.stack(frames, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return frames, targets, lengths

def collate_fn_sqn(data):
    """ eates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    frames, captions, seqs = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    frames = torch.stack(frames, 0)
    seqs = torch.stack(seqs, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return frames, targets, seqs, lengths

def get_box_center(box):
    y1 = box[0]
    y2 = box[2]
    x1 = box[1]
    x2 = box[3]
    return (x1 + x2)/2, (y1 + y2)/2


