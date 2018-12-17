import os
import sys
import functools

import imageio
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset
from torch import Tensor
from torch.autograd import Variable
import random
import logging
import nltk
from vocabulary import Vocabulary
from ipdb import set_trace
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import time
from pyquaternion import Quaternion
from torchvision.transforms import ColorJitter, RandomResizedCrop, Compose, ToTensor, ToPILImage

from builder_utils import distance, view_image, write_to_csv, ensure_folder, resize_frame, write_video, \
                read_video, read_extracted_rcnn_results, read_caption, ls_directories, ls, ls_unparsed_txt, ls_npy, \
                ls_txt, ls_view, read_extracted_video, Logger, ls_extracted, crop_box, crop_uniform_box, get_box_center
sys.path.append('/home/msieb/projects/general-utils')
from rot_utils import create_rot_from_vector, rotationMatrixToEulerAngles, isRotationMatrix, eulerAnglesToRotationMatrix
from plot_utils import concat_frames_nosave

##### PATHS #####
sys.path.append('/home/max/projects/gps-lfd') 
sys.path.append('/home/msieb/projects/gps-lfd')
from config_server import Config_Isaac_Server as Config, Multi_Camera_Config as Camera_Config # Import approriate config
conf = Config()

# Camera Config
cam_conf = Camera_Config()
VIEW_PARAMS = cam_conf.VIEW_PARAMS
PROJ_PARAMS = cam_conf.PROJ_PARAMS
ROT_MATRICES = cam_conf.ROT_MATRICES
#sys.path.append('/home/msieb/projects/Mask_RCNN/samples')
#from baxter.baxter import BaxterConfig, InferenceConfig

OFFSET = 0
tt = ToTensor()
cj = ColorJitter(0.2, 0.1, 0,0)
rc = RandomResizedCrop(size=299, scale=(0.8,1.0), ratio=(.8,1.2))
tp = ToPILImage()

content_transform = Compose([
tp,
rc,
cj



 ])

class SingleViewTripletBuilder(object):
    def __init__(self, view, video_directory, image_size, cli_args, sample_size=500):
        self.frame_size = image_size
        self.view = view
        self._read_video_dir(video_directory)
        #self._read_extracted_video_dir(video_directory)
        self._count_frames()
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.positive_frame_margin = 3
        self.negative_frame_margin = 10 
        self.video_index = 0
        self.cli_args = cli_args
        self.sample_size = sample_size

    def _read_video_dir(self, video_directory):
        self._video_directory = video_directory
        filenames = ls(video_directory)
        self.video_paths = [os.path.join(self._video_directory, f) for f in filenames]
        self.video_count = len(self.video_paths)

    def _read_extracted_video_dir(self, video_directory):
        self._video_directory = video_directory
        filenames = ls_extracted(video_directory)
        # self.video_paths = [os.path.join(self._video_directory, f.split('.mp4')[0]) for f in filenames]
        self.video_paths = [os.path.join(self._video_directory, f) for f in filenames if not f.endswith('.mp4') and 'debug' not in f]
        self.video_count = len(self.video_paths)

    def _count_frames(self):
        frame_lengths = np.array([len(imageio.read(p)) for p in self.video_paths])
        self.frame_lengths = frame_lengths
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    def _count_extracted_frames(self):
        frame_lengths = np.array([len(os.listdir(p)) for p in self.video_paths if p.endswith('.jpg')])
        self.frame_lengths = frame_lengths
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    @functools.lru_cache(maxsize=1)
    def get_video(self, index):
        return read_video(self.video_paths[index], self.frame_size)

    def sample_triplet(self, snap):
        anchor_index = self.sample_anchor_frame_index()
        positive_index = self.sample_positive_frame_index(anchor_index)
        negative_index = self.sample_negative_frame_index(anchor_index)
        anchor_frame = snap[anchor_index]
        positive_frame = snap[positive_index]
        negative_frame = snap[negative_index]

        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame))

    def build_set(self):
        triplets = []
        triplets = torch.Tensor(self.sample_size, 3, 3, *self.frame_size)
        # print("Create triplets from {}".format(self.video_paths[self.video_index]))
        snap = self.get_video(self.video_index)
        for i in range(0, self.sample_size):
            anchor_frame, positive_frame, negative_frame = self.sample_triplet(snap)
            triplets[i, 0, :, :, :] = anchor_frame
            triplets[i, 1, :, :, :] = positive_frame
            triplets[i, 2, :, :, :] = negative_frame
        self.video_index = (self.video_index + 1) % self.video_count

        # Second argument is labels. Not used.
        return TensorDataset(triplets, torch.zeros(triplets.size()[0]))

    def sample_anchor_frame_index(self):
        arange = np.arange(0, self.frame_lengths[self.video_index])
        return np.random.choice(arange)

    def sample_positive_frame_index(self, anchor_index):
        lower_bound = max(0, anchor_index - self.positive_frame_margin)
        range1 = np.arange(lower_bound, anchor_index)
        upper_bound = min(self.frame_lengths[self.video_index] - 1, anchor_index + self.positive_frame_margin)
        range2 = np.arange(anchor_index + 1, upper_bound)
        return np.random.choice(np.concatenate([range1, range2]))

    def negative_frame_indices(self, anchor_index):
        video_length = self.frame_lengths[self.video_index]
        lower_bound = 0
        upper_bound = max(0, anchor_index - self.negative_frame_margin)
        range1 = np.arange(lower_bound, upper_bound)
        lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
        upper_bound = video_length 
        range2 = np.arange(lower_bound, upper_bound)
        return np.concatenate([range1, range2])

    def sample_negative_frame_index(self, anchor_index):
        return np.random.choice(self.negative_frame_indices(anchor_index))

class MultiViewTripletBuilder(object):
    def __init__(self, n_views, video_directory, image_size, cli_args, sample_size=500):
        self.frame_size = image_size
        self.n_views = n_views
        self._read_video_dir(video_directory)

        self._count_frames()
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.negative_frame_margin = 10 
        self.sequence_index = 0
        self.cli_args = cli_args
        self.sample_size = sample_size

    def _read_video_dir(self, video_directory):
        self._video_directory = video_directory
        filenames = ls(video_directory)
        self.video_paths = [os.path.join(self._video_directory, f) for f in filenames if f.endswith('.mp4')]
        # for path in self.video_paths:
        #     print(path)
        self.sequence_count = int(len(self.video_paths) / self.n_views)

    def _count_frames(self):
        frame_lengths = np.array([len(imageio.read(p)) for p in self.video_paths])
        self.frame_lengths = frame_lengths - OFFSET
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    @functools.lru_cache(maxsize=1)
    def get_videos(self, index):
        views = []
        debug_paths = []
        for i in range(self.n_views):
            views.append(read_video(self.video_paths[index + i], self.frame_size))
            debug_paths.append(self.video_paths[index + i])
        return views, debug_paths

    def sample_triplet(self, snaps):
        loaded_sample = False
        while not loaded_sample:

            try:
                anchor_index = self.sample_anchor_frame_index()
                positive_index = anchor_index
                negative_index = self.sample_negative_frame_index(anchor_index)
                loaded_sample = True
            except:
                pass
                # print("Error loading video - sequence index: ", self.sequence_index)
                # print("video lengths: ", [len(snaps[i]) for i in range(0, len(snaps))])
                # print("Maybe margin too high")
        # random sample anchor view,and positive view
        view_set = set(range(self.n_views))
        anchor_view = np.random.choice(np.array(list(view_set)))
        view_set.remove(anchor_view)
        positive_view = np.random.choice(np.array(list(view_set)))
        negative_view = anchor_view # negative example comes from same view INQUIRE TODO

        anchor_frame = snaps[anchor_view][anchor_index]
        positive_frame = snaps[positive_view][positive_index]
        negative_frame = snaps[negative_view][negative_index]
        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame))

    def build_set(self):
        triplets = []
        triplets = torch.Tensor(self.sample_size, 3, 3, *self.frame_size)
        snaps, debug_paths = self.get_videos(self.sequence_index * self.n_views)
        for i in range(0, self.sample_size):
            anchor_frame, positive_frame, negative_frame = self.sample_triplet(snaps)
            triplets[i, 0, :, :, :] = anchor_frame
            triplets[i, 1, :, :, :] = positive_frame
            triplets[i, 2, :, :, :] = negative_frame

        self.sequence_index = (self.sequence_index + 1) % self.sequence_count
        
        # Second argument is labels. Not used.
        return TensorDataset(triplets, torch.zeros(triplets.size()[0]))

    def sample_anchor_frame_index(self):
        # if self.frame_lengths[self.sequence_index * self.n_views] == 1:
        #     return 0
        # else:
        arange = np.arange(0, self.frame_lengths[self.sequence_index * self.n_views])
        return np.random.choice(arange)

    # def samplfe_positive_frame_index(self, anchor_inde4x):

    #     upper_bound = min(self.frame_lengths[self.sequence_index * self.n_views + 1] - 1, anchor_index)
    #     return upper_bound # in case video has less frames than anchor video

    def negative_frame_indices(self, anchor_index):
        video_length = self.frame_lengths[self.sequence_index * self.n_views]
        lower_bound = 0
        upper_bound = max(0, anchor_index - self.negative_frame_margin)
        range1 = np.arange(lower_bound, upper_bound)
        lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
        upper_bound = video_length
        range2 = np.arange(lower_bound, upper_bound)
        return np.concatenate([range1, range2])

    def sample_negative_frame_index(self, anchor_index):
        return np.random.choice(self.negative_frame_indices(anchor_index))

class PoseMultiViewTripletBuilder(MultiViewTripletBuilder):
    def __init__(self, n_views, video_directory, image_size, cli_args, sample_size=500):
        super(PoseMultiViewTripletBuilder, self).__init__(n_views, video_directory, image_size, cli_args, sample_size)
    
    @functools.lru_cache(maxsize=1)
    def get_poses(self, index):
        views = []
        for i in range(self.n_views):
            views.append(np.load(self.video_paths[index + i].split('.mp4')[0]+'.npy')[:, -4:])
        return views

    def sample_triplet(self, snaps, poses):
        loaded_sample = False
        while not loaded_sample:

            try:
                anchor_index = self.sample_anchor_frame_index()
                positive_index = anchor_index
                negative_index = self.sample_negative_frame_index(anchor_index)
                loaded_sample = True
            except:
                pass
                # print("Error loading video - sequence index: ", self.sequence_index)
                # print("video lengths: ", [len(snaps[i]) for i in range(0, len(snaps))])
                # print("Maybe margin too high")
        # random sample anchor view,and positive view
        view_set = set(range(self.n_views))
        anchor_view = np.random.choice(np.array(list(view_set)))
        view_set.remove(anchor_view)
        positive_view = np.random.choice(np.array(list(view_set)))
        negative_view = anchor_view # negative example comes from same view INQUIRE TODO

        anchor_frame = snaps[anchor_view][anchor_index]
        positive_frame = snaps[positive_view][positive_index]
        negative_frame = snaps[negative_view][negative_index]
        # what shape has pose? T x 7?
        anchor_pose= poses[anchor_view][anchor_index]
        positive_pose= poses[positive_view][positive_index]
        negative_pose= poses[negative_view][negative_index]

        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame), torch.Tensor(anchor_pose), torch.Tensor(positive_pose), torch.Tensor(negative_pose))

    def build_set(self):
        triplets = []
        triplets = torch.Tensor(self.sample_size, 3, 3, *self.frame_size)
        pose_triplets = torch.Tensor(self.sample_size, 3, 4)
        snaps, debug_paths = self.get_videos(self.sequence_index * self.n_views)
        #print("building set from video sequence, loaded paths: {}".format(debug_paths))
        for i in range(0, self.sample_size):
            poses = self.get_poses(self.sequence_index * self.n_views)
            anchor_frame, positive_frame, negative_frame, \
                       anchor_pose, positive_pose, negative_pose = self.sample_triplet(snaps, poses)
            triplets[i, 0, :, :, :] = anchor_frame
            triplets[i, 1, :, :, :] = positive_frame
            triplets[i, 2, :, :, :] = negative_frame
            pose_triplets[i, 0, :] = anchor_pose
            pose_triplets[i, 1, :] = positive_pose
            pose_triplets[i, 2, :] = negative_pose
        self.sequence_index = (self.sequence_index + 1) % self.sequence_count
        # Second argument is labels. Not used.
        return TensorDataset(triplets, pose_triplets)

class TwoViewBuilder(MultiViewTripletBuilder):
    def __init__(self, n_views, video_directory, image_size, cli_args, sample_size=500, n_seqs=10000):
        self.frame_size = image_size
        self.n_views = n_views
        self.n_seqs = n_seqs
        self._read_video_dir(video_directory)

        self._count_frames()
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.negative_frame_margin = 10 
        self.sequence_index = 0
        self.cli_args = cli_args
        self.sample_size = sample_size
        self.rots = self.get_rot() 
        
    def _read_video_dir(self, video_directory):
        self._video_directory = video_directory
        filenames = ls(video_directory)
        self.video_paths = [os.path.join(self._video_directory, f) for f in filenames if f.endswith('.mp4') and \
                    int(f.split('view')[-1].split('.mp4')[0]) < self.n_views and \
                    int(f.split('_')[0])  < self.n_seqs]
        #print(self.video_paths)
        print("Got {} video paths".format(len(self.video_paths)))
        # for path in self.video_paths:
        #     print(path)
        self.sequence_count = int(len(self.video_paths) / self.n_views)

    @functools.lru_cache(maxsize=1)
    #def get_rot(self):
    #    views = []
    #    for i in range(self.n_views):
    #        views.append(ROT_MATRICES[i])
    #    return views
    def get_rot(self):
        views = np.load(os.path.join(self._video_directory, "{}_camparams.npy".format(self.sequence_index)))
        return views
    
    def sample_triplet(self, snaps, rot):
        anchor_frames = np.zeros((2, 3, 299, 299))
        anchor_index = self.sample_anchor_frame_index()
        positive_index = anchor_index
        # negative_index = self.sample_negative_frame_index(anchor_index)
        # random sample anchor view,and positive view
        view_set = set(range(self.n_views))
        anchor_view = np.random.choice(np.array(list(view_set)))
        view_set.remove(anchor_view)
        positive_view = np.random.choice(np.array(list(view_set)))
        #negative_view = anchor_view # negative example comes from same view INQUIRE TODO
        anchor_frames[0] = snaps[anchor_view][anchor_index]
        anchor_frames[1] = snaps[positive_view][positive_index]
        #negative_frame = snaps[negative_view][negative_index]
        # what shape has pose? T x 7?
        delta_rot = rot[positive_view].dot(rot[anchor_view].T)
        #delta_euler = rotationMatrixToEulerAngles(delta_rot)
        #delta_rot = np.reshape(delta_rot[:, :-1], -1) # only predict first two columns
        assert isRotationMatrix(delta_rot)
        return (torch.Tensor(anchor_frames), torch.Tensor(delta_rot))

    def build_set(self):
        frames = torch.Tensor(self.sample_size, 2, 3, *self.frame_size)
        deltas_rot = torch.Tensor(self.sample_size, 3, 3)
        snaps, debug_paths = self.get_videos(self.sequence_index * self.n_views)
        #print("building set from video sequence, loaded paths: {}".format(debug_paths))
        for i in range(0, self.sample_size):
            anchor_frame, delta_rot = self.sample_triplet(snaps, self.rots)

            frames[i, 0, :, :, :] = anchor_frame[0]
            frames[i, 1, :, :, :] = anchor_frame[1]
            deltas_rot[i] = delta_rot
        self.sequence_index = (self.sequence_index + 1) % self.sequence_count
        # Second argument is labels. Not used.
        return TensorDataset(frames, deltas_rot)

class TwoViewQuaternionBuilder(TwoViewBuilder):
    def __init__(self, n_views, video_directory, image_size, cli_args, sample_size=500, n_seqs=10000):
        super(TwoViewQuaternionBuilder, self).__init__(n_views, video_directory, image_size, cli_args, sample_size, n_seqs)
    
    def sample_triplet(self, snaps, rot):
        anchor_frames = np.zeros((2, 3, 299, 299))
        anchor_index = self.sample_anchor_frame_index()
        positive_index = anchor_index
        # negative_index = self.sample_negative_frame_index(anchor_index)
        # random sample anchor view,and positive view
        # Remove anchor view
        # view_set = set(range(self.n_views))
        # anchor_view = np.random.choice(np.array(list(view_set)))
        # view_set.remove(anchor_view)
        # positive_view = np.random.choice(np.array(list(view_set)))

        # Dont remove anchor view
        view_set = range(self.n_views)
        anchor_view = np.random.choice(np.array(view_set))
        positive_view = np.random.choice(np.array(view_set))
        
        #seed = random.randint(0,2**32)
        #random.seed(seed)
        #anchor_frames[0] = tt(cj(rc(tp((np.uint8(255*snaps[anchor_view][anchor_index]))))))
        #random.seed(seed)
        #anchor_frames[0] = tt(cj(rc(tp((np.uint8(255*snaps[anchor_view][anchor_index]))))))
        anchor_frames[0] = snaps[anchor_view][anchor_index]
        anchor_frames[1] = snaps[positive_view][positive_index]
        #negative_frame = snaps[negative_view][negative_index]
        # what shape has pose? T x 7?
        delta_rot = rot[positive_view].dot(rot[anchor_view].T)
        delta_quat = Quaternion(matrix=delta_rot)
        delta_quat = delta_quat.elements
        delta_quat /= np.linalg.norm(delta_quat)
        #delta_euler = rotationMatrixToEulerAngles(delta_rot)
        #delta_rot = np.reshape(delta_rot[:, :-1], -1) # only predict first two columns
        assert isRotationMatrix(delta_rot)
        return (torch.Tensor(anchor_frames), torch.Tensor(delta_quat))

    def build_set(self):
        frames = torch.Tensor(self.sample_size, 2, 3, *self.frame_size)
        deltas_quat = torch.Tensor(self.sample_size, 4)
        snaps, debug_paths = self.get_videos(self.sequence_index * self.n_views)
        #print("building set from video sequence, loaded paths: {}".format(debug_paths))
        for i in range(0, self.sample_size):
            anchor_frame, delta_quat = self.sample_triplet(snaps, self.rots)
            frames[i, 0, :, :, :] = anchor_frame[0]
            frames[i, 1, :, :, :] = anchor_frame[1]
            deltas_quat[i] = delta_quat
        self.sequence_index = (self.sequence_index + 1) % self.sequence_count
        # Second argument is labels. Not used.
        return TensorDataset(frames, deltas_quat)

# class MultiViewQuaternionBuilder(TwoViewQuaternionBuilder):
#     def __init__(self, n_views, video_directory, image_size, cli_args, sample_size=500, n_seqs=10000):
#         super(MultiViewQuaternionBuilder, self).__init__(n_views, video_directory, image_size, cli_args, sample_size, n_seqs)
    
#     def sample_triplet(self, snaps, rot):
#         anchor_frames = np.zeros((3, 3, 299, 299))
#         anchor_index = self.sample_anchor_frame_index()
#         positive_index = anchor_index
#         # negative_index = self.sample_negative_frame_index(anchor_index)
#         # random sample anchor view,and positive view
#         #Remove anchor view
#         view_set = set(range(self.n_views))
#         anchor_view = np.random.choice(np.array(list(view_set)))
#         view_set.remove(anchor_view)
#         positive_view = np.random.choice(np.array(list(view_set)))

#         # Dont remove anchor view
#         # view_set = range(self.n_views)
#         # anchor_view = np.random.choice(np.array(view_set))
#         # positive_view = np.random.choice(np.array(view_set))
#         # anchor_frames[0] = snaps[anchor_view][anchor_index]
#         # anchor_frames[1] = snaps[positive_view][positive_index]

#         #negative_frame = snaps[negative_view][negative_index]
#         # what shape has pose? T x 7?
#         delta_rot = rot[positive_view].dot(rot[anchor_view].T)
#         delta_quat = Quaternion(matrix=delta_rot)
#         delta_quat = delta_quat.elements
#         delta_quat /= np.linalg.norm(delta_quat)
#         #delta_euler = rotationMatrixToEulerAngles(delta_rot)
#         #delta_rot = np.reshape(delta_rot[:, :-1], -1) # only predict first two columns
#         assert isRotationMatrix(delta_rot)
#         return (torch.Tensor(anchor_frames), torch.Tensor(delta_quat))

    def build_set(self):
        frames = torch.Tensor(self.sample_size, 2, 3, *self.frame_size)
        deltas_quat = torch.Tensor(self.sample_size, 4)
        snaps, debug_paths = self.get_videos(self.sequence_index * self.n_views)
        #print("building set from video sequence, loaded paths: {}".format(debug_paths))
        for i in range(0, self.sample_size):
            anchor_frame, delta_quat = self.sample_triplet(snaps, self.rots)
            frames[i, 0, :, :, :] = anchor_frame[0]
            frames[i, 1, :, :, :] = anchor_frame[1]
            deltas_quat[i] = delta_quat
        self.sequence_index = (self.sequence_index + 1) % self.sequence_count
        # Second argument is labels. Not used.
        return TensorDataset(frames, deltas_quat)


class OneViewQuaternionBuilder(TwoViewQuaternionBuilder):
    def __init__(self, n_views, video_directory, image_size, cli_args, sample_size=500, n_seqs=10000):
        super(OneViewQuaternionBuilder, self).__init__(n_views, video_directory, image_size, cli_args, sample_size, n_seqs)
    
    def sample_triplet(self, snaps, rot):
        anchor_index = self.sample_anchor_frame_index()
        positive_index = anchor_index
        
        # random sample anchor view,and positive view
        view_set = set(range(self.n_views))
        anchor_view = np.random.choice(np.array(list(view_set)))
        anchor_frame = snaps[anchor_view][anchor_index]
        
        anchor_frame = np.transpose(np.asarray(content_transform(np.uint8(np.transpose(255*anchor_frame, [1,2,0])))) / 255.0, [2,0,1])
        #seed = random.randint(0,2**32)
        #random.seed(seed)
        #test2 = np.transpose(np.asarray(cj(rc(tp(np.uint8(np.transpose(255*anchor_frame, [1,2,0])))))) / 255.0, [2,0,1])
        #random.seed(seed)
        #test = np.transpose(np.asarray(content_transform(np.uint8(np.transpose(255*anchor_frame, [1,2,0])))) / 255.0, [2,0,1])
        # what shape has pose? T x 7?
        rot = rot[anchor_view].T
        quat = Quaternion(matrix=rot)
        quat = quat.elements
        quat /= np.linalg.norm(quat)
        #delta_euler = rotationMatrixToEulerAngles(delta_rot)
        #delta_rot = np.reshape(delta_rot[:, :-1], -1) # only predict first two columns
        assert isRotationMatrix(rot)
        return (torch.Tensor(anchor_frame), torch.Tensor(quat))

    def build_set(self):
        frames = torch.Tensor(self.sample_size, 3, *self.frame_size)
        quats = torch.Tensor(self.sample_size, 4)
        snaps, debug_paths = self.get_videos(self.sequence_index * self.n_views)
        #print("building set from video sequence, loaded paths: {}".format(debug_paths))
        for i in range(0, self.sample_size):
            anchor_frame, quat = self.sample_triplet(snaps, self.rots)
            frames[i, :, :, :] = anchor_frame
            quats[i] = quat
        self.sequence_index = (self.sequence_index + 1) % self.sequence_count
        # Second argument is labels. Not used.
        return TensorDataset(frames, quats) 

class TwoViewEulerBuilder(TwoViewBuilder):
    def __init__(self, n_views, video_directory, image_size, cli_args, sample_size=500):
        super(TwoViewEulerBuilder, self).__init__(n_views, video_directory, image_size, cli_args, sample_size)
    

    def sample_triplet(self, snaps, rot):
        anchor_frames = np.zeros((2, 3, 299, 299))
        anchor_index = self.sample_anchor_frame_index()
        positive_index = anchor_index
        # negative_index = self.sample_negative_frame_index(anchor_index)
        # random sample anchor view,and positive view
        view_set = set(range(self.n_views))
        anchor_view = np.random.choice(np.array(list(view_set)))
        view_set.remove(anchor_view)
        positive_view = np.random.choice(np.array(list(view_set)))
        #negative_view = anchor_view # negative example comes from same view INQUIRE TODO
        #print("anchor view: {}, pos view: {}".format(anchor_view, positive_view))
        anchor_frames[0] = snaps[anchor_view][anchor_index]
        anchor_frames[1] = snaps[positive_view][positive_index]
        #negative_frame = snaps[negative_view][negative_index]
        # what shape has pose? T x 7?
        delta_rot = rot[positive_view].dot(rot[anchor_view].T)
        assert isRotationMatrix(delta_rot)

        #delta_rot = np.reshape(delta_rot[:, :-1], -1) # only predict first two columns
        return (torch.Tensor(anchor_frames), torch.Tensor(delta_euler_reparam))

    def build_set(self):
        frames = torch.Tensor(self.sample_size, 2, 3, *self.frame_size)
        deltas_euler_reparam = torch.Tensor(self.sample_size, 6)
        snaps, debug_paths = self.get_videos(self.sequence_index * self.n_views)
        #print("building set from video sequence, loaded paths: {}".format(debug_paths))
        for i in range(0, self.sample_size):
            anchor_frame, delta_euler_reparam = self.sample_triplet(snaps, self.rots)

            frames[i, 0, :, :, :] = anchor_frame[0]
            frames[i, 1, :, :, :] = anchor_frame[1]

            deltas_euler_reparam[i] = delta_euler_reparam
        self.sequence_index = (self.sequence_index + 1) % self.sequence_count
        # Second argument is labels. Not used.
        return TensorDataset(frames, deltas_euler_reparam)

class SingleViewDepthTripletBuilder(SingleViewTripletBuilder):
    def __init__(self, view, video_directory, depth_video_directory, image_size, cli_args, sample_size=500):
        super(SingleViewDepthTripletBuilder, self).__init__(view, video_directory, image_size, cli_args, sample_size)
        self._read_depth_video_dir(depth_video_directory)

    def _read_depth_video_dir(self, depth_video_directory):
        self._depth_video_directory = depth_video_directory
        filenames = ls_extracted(depth_video_directory)
        self.depth_video_paths = [os.path.join(self._depth_video_directory, f) for f in filenames]
        self.depth_video_count = len(self.depth_video_paths)

    def _read_extracted_depth_video_dir(self, depth_video_directory):
        self._depth_video_directory = depth_video_directory
        filenames = ls_extracted(depth_video_directory)
        self.depth_video_paths = [os.path.join(self._depth_video_directory, f.split('.mp4')[0]) for f in filenames]
        self.depth_video_paths = [os.path.join(self._depth_video_directory, f) for f in filenames if not f.endswith('.mp4') and 'debug' not in f]
        self.depth_video_count = len(self.depth_video_paths)

    def sample_triplet(self, snap, snap_depth):
        anchor_index = self.sample_anchor_frame_index()
        positive_index = self.sample_positive_frame_index(anchor_index)
        negative_index = self.sample_negative_frame_index(anchor_index)
        try:
            anchor_frame = np.concatenate([snap[anchor_index], snap_depth[anchor_index][None, 0]], axis=0)
            positive_frame = np.concatenate([snap[positive_index], snap_depth[positive_index][None, 0]], axis=0)
            negative_frame = np.concatenate([snap[negative_index], snap_depth[negative_index][None, 0]], axis=0)
        except:
            print(self.video_index)
            print(len(snap))
            print(len(snap_depth))
        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame))

    def build_set(self):
        triplets = []
        triplets = torch.Tensor(self.sample_size, 3, 4, *self.frame_size)
        for i in range(0, self.sample_size):
            snap = self.get_video(self.video_index)
            depth_snap = self.get_depth_video(self.video_index)
            anchor_frame, positive_frame, negative_frame = self.sample_triplet(snap, depth_snap)
            triplets[i, 0, :, :, :] = anchor_frame
            triplets[i, 1, :, :, :] = positive_frame
            triplets[i, 2, :, :, :] = negative_frame
        #print(self.video_index)

        self.video_index = (self.video_index + 1) % self.video_count
        # Second argument is labels. Not used.
        return TensorDataset(triplets, torch.zeros(triplets.size()[0]))

    def get_depth_video(self, index):
        return read_video(self.depth_video_paths[index], self.frame_size)

class SingleViewDepthTripletRCNNBuilder(SingleViewDepthTripletBuilder):
    def __init__(self, view, video_directory, depth_video_directory, image_size, cli_args, sample_size=500):
        self.frame_size = image_size
        self.view = view
        self._read_extracted_video_dir(video_directory)
        self._count_extracted_frames()
        self._read_extracted_depth_video_dir(depth_video_directory)
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.positive_frame_margin = 2
        self.negative_frame_margin = 4
        self.video_index = 0
        self.cli_args = cli_args
        self.sample_size = sample_size        

        self.inference_config = InferenceConfig()

        self.transform = None
        self.class_names = self.inference_config.CLASS_NAMES
        self.target_ids = self.inference_config.TARGET_IDS

    def sample_triplet(self, snap, snap_depth, snap_feature):
        def _smooth(image):
            return ndi.filters.gaussian_filter(image, (7, 7, 0), order=0)

        anchor_index = self.sample_anchor_frame_index(snap)
        positive_index = self.sample_positive_frame_index(anchor_index, snap)
        negative_index = self.sample_negative_frame_index(anchor_index, snap)
        # try:
        anchor_frame = np.concatenate([snap[anchor_index], _smooth(snap_depth[anchor_index][None, 0])], axis=0)
        positive_frame = np.concatenate([snap[positive_index], _smooth(snap_depth[positive_index][None, 0])], axis=0)
        negative_frame = np.concatenate([snap[negative_index], _smooth(snap_depth[negative_index][None, 0])], axis=0)
        anchor_feature = snap_feature[anchor_index]
        positive_feature = snap_feature[positive_index]
        negative_feature = snap_feature[negative_index]

        # except:
        #     set_trace()
        #     print(self.video_index)
        #     print(len(snap))
        #     print(len(snap_depth))


        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame), torch.Tensor(anchor_feature), torch.Tensor(positive_feature),
            torch.Tensor(negative_feature))

    def sample_fullframe_triplet(self, snap, snap_depth):
        def _smooth(image):
            return ndi.filters.gaussian_filter(image, (7, 7, 0), order=0)

        anchor_index = self.sample_anchor_frame_index(snap)
        positive_index = self.sample_positive_frame_index(anchor_index, snap)
        negative_index = self.sample_negative_frame_index(anchor_index, snap)
        # try:



        anchor_frame = np.concatenate([resize_frame(snap[anchor_index], (299, 299)), _smooth(resize_frame(snap_depth[anchor_index], (299, 299))[None, 0])], axis=0)
        positive_frame = np.concatenate([resize_frame(snap[positive_index], (299, 299)), _smooth(resize_frame(snap_depth[positive_index], (299, 299))[None, 0])], axis=0)
        negative_frame = np.concatenate([resize_frame(snap[negative_index], (299, 299)), _smooth(resize_frame(snap_depth[negative_index], (299, 299))[None, 0])], axis=0)
        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame))


    def build_set(self, full_frame=False):
        # build image triplet item
        # build caption item
        triplets = []
        triplets = torch.Tensor(self.sample_size, 3, 4, *self.frame_size)
        triplets_features = torch.Tensor(self.sample_size, 3, 256, 7, 7)
        snap = self.get_extracted_video(self.video_index)
        depth_snap = self.get_extracted_depth_video(self.video_index)
        results_snap = self.get_extracted_rcnn_results(self.video_index)
        object_snaps, object_depth_snaps, object_feature_snaps = self._make_object_snaps(snap, depth_snap, results_snap)

        if not full_frame:
            i = 0
            while i < self.sample_size:
                # print("sample ", i)


                for object_snap, object_depth_snap, object_feature_snap in zip(object_snaps.values(), object_depth_snaps.values(), object_feature_snaps.values()):
                    anchor_frame, positive_frame, negative_frame, \
                        anchor_feature, positive_feature, \
                        negative_feature = self.sample_triplet(object_snap, object_depth_snap, object_feature_snap)
                    triplets[i, 0, :, :, :] = anchor_frame
                    triplets[i, 1, :, :, :] = positive_frame
                    triplets[i, 2, :, :, :] = negative_frame
                    triplets_features[i, 0, :, :, :] = anchor_feature
                    triplets_features[i, 1, :, :, :] = positive_feature
                    triplets_features[i, 2, :, :, :] = negative_feature
                    i += 1
        else:
            for i in range(0, self.sample_size):
                # print("sample ", i)

                anchor_frame, positive_frame, negative_frame = self.sample_fullframe_triplet(snap, depth_snap)       
                triplets[i, 0, :, :, :] = anchor_frame
                triplets[i, 1, :, :, :] = positive_frame
                triplets[i, 2, :, :, :] = negative_frame
                triplets_features[i, 0, :, :, :] = torch.zeros((256, 7, 7))
                triplets_features[i, 1, :, :, :] = torch.zeros((256, 7, 7))
                triplets_features[i, 2, :, :, :] = torch.zeros((256, 7, 7))            

        print(self.video_index)   
        # active_label = self.vocab(label[-4])
        # passive_label = self.vocab(label[-2])
        # Convert caption (string) to word ids.
        self.video_index = (self.video_index + 1) % self.video_count
        return TensorDataset(triplets, triplets_features)

    def _make_object_snaps(self, snap, depth_snap, results_snap):
        object_snaps = {key: [] for key in self.target_ids}
        object_depth_snaps = {key: [] for key in self.target_ids}
        object_feature_snaps = {key: [] for key in self.target_ids}

        for i, image in enumerate(snap):
            r = results_snap[i]
            encountered_ids = []
            depth_image = depth_snap[i]
            for i, box in enumerate(r['rois']):
                class_id = r['class_ids'][i]
                if class_id not in self.target_ids or class_id in encountered_ids:
                    continue
                box_center = get_box_center(box)
                # if box_center[1] < 150:
                #     continue
                encountered_ids.append(class_id)
                cropped = crop_uniform_box(image, box, height_scaling=150, width_scaling=150)
                # cropped = cv2.resize(cropped, (299, 299))
                # cropped = np.transpose(cropped, [2, 0, 1])
                cropped = resize_frame(cropped, (299, 299))
                object_snaps[class_id].append(cropped)

                depth_cropped = crop_uniform_box(depth_image, box, height_scaling=150, width_scaling=150)
                depth_cropped = resize_frame(depth_cropped, (299, 299))
                object_depth_snaps[class_id].append(depth_cropped)
                object_feature_snaps[class_id].append(np.transpose(r['roi_features'][i], [2, 0, 1]))
        # # FOR DEBUG #
        # for obj_snap in object_snaps.values():
        #     if len(obj_snap) < 6:
        #         continue
        #     concated_frames =concat_frames_nosave(obj_snap)

        #     cut_off = min(concated_frames.shape[2], 6*640)
        #     print(concated_frames.shape)            
        #    # fig = plt.figure()

        #   # ax.axis('tight')
        #     concat_img = np.transpose(concated_frames[:, :, :cut_off], [1,2,0])
        #     fig = plt.figure(figsize=(30, 3))
        #     ax = fig.add_subplot(111)
        #     plt.imshow(concat_img)
        #     fig.subplots_adjust(hspace=0, wspace=0)
        #     ax.set_axis_off()
        #     fig.savefig(os.path.join('/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/humanoids18/object_centric_tubes/tower', str(int(time.time())) + '.jpg'))
        #     plt.close(fig)
        #     # plt.axis('tight')
        #     # plt.show()
        #     # set_trace()
        #     # plt.imsave(os.path.join('/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/humanoids18/object_centric_tubes/mugs', str(int(time.time())) + '.jpg'), concat_img)
        #############
        invalid_keys = [] # not enough detections, dont use that tube
        for key, val in object_snaps.items():
            if len(val) < 8:
                invalid_keys.append(key)
        for invalid_key in invalid_keys:
            del object_snaps[invalid_key]
            del object_depth_snaps[invalid_key]
            del object_feature_snaps[invalid_key]
        return object_snaps, object_depth_snaps, object_feature_snaps


    def _read_label_dir(self, label_directory):
        self._label_directory = label_directory
        filenames = ls_txt(label_directory)
        self.label_paths = [os.path.join(self._label_directory, f) for f in filenames]
        
    def get_caption(self, index):
        caption = read_caption(self.caption_paths[index])
        return caption

    def get_extracted_video(self, index):
        return read_extracted_video(self.video_paths[index], self.frame_size)

    def get_extracted_depth_video(self, index):
        return read_extracted_video(self.depth_video_paths[index], self.frame_size)

    def get_extracted_rcnn_results(self, index):
        return read_extracted_rcnn_results(self.video_paths[index], self.frame_size)

    def concat_frames(self, images):
        for i, img in enumerate(images):
            if i == 0:
                concat_img = img
                continue
            else:
                concat_img = np.concatenate([concat_img, img], axis=1)
        plt.imshow(concat_img)
        plt.show()
        return concat_img

    def sample_anchor_frame_index(self, video):
        arange = np.arange(0, len(video))
        return np.random.choice(arange)

    def sample_positive_frame_index(self, anchor_index, video):
        video_length = len(video)
        lower_bound = max(0, anchor_index - self.positive_frame_margin)
        range1 = np.arange(lower_bound, anchor_index)
        upper_bound = min(video_length - 1, anchor_index + self.positive_frame_margin)
        range2 = np.arange(anchor_index + 1, upper_bound)
        return np.random.choice(np.concatenate([range1, range2]))

    def negative_frame_indices(self, anchor_index, video):
        video_length = len(video)
        lower_bound = 0
        upper_bound = max(0, anchor_index - self.negative_frame_margin)
        range1 = np.arange(lower_bound, upper_bound)
        lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
        upper_bound = video_length 
        range2 = np.arange(lower_bound, upper_bound)
        return np.concatenate([range1, range2])

    def sample_negative_frame_index(self, anchor_index, video):
        return np.random.choice(self.negative_frame_indices(anchor_index, video))

class SingleViewDepthTripletExtractedBuilder(SingleViewDepthTripletBuilder):
    def __init__(self, view, video_directory, depth_video_directory, image_size, cli_args, sample_size=500):
        self.frame_size = image_size
        self.view = view
        self._read_extracted_video_dir(video_directory)
        self._count_extracted_frames()
        self._read_extracted_depth_video_dir(depth_video_directory)
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.positive_frame_margin = 2
        self.negative_frame_margin = 4
        self.video_index = 0
        self.cli_args = cli_args
        self.sample_size = sample_size        

        self.transform = None
        self.class_names = ['BG', 'blue_ring', 'green_ring', 'yellow_ring', 'tower', 'hand', 'robot']
        self.target_ids = sorted([1, 2, 4])

    def sample_triplet(self, snap, snap_depth):
        def _smooth(image):
            return ndi.filters.gaussian_filter(image, (7, 7, 0), order=0)

        anchor_index = self.sample_anchor_frame_index(snap)
        positive_index = self.sample_positive_frame_index(anchor_index, snap)
        negative_index = self.sample_negative_frame_index(anchor_index, snap)
        # try:
        cropped = resize_frame(cropped, (299, 299))
        depth_cropped = resize_frame(snap_depth[anchor_index], (299, 299))
        depth_cropped = np.transpose(depth_cropped, [2, 0, 1])
        anchor_frame = np.concatenate([cropped, _smooth(depth_cropped[None, 0])], axis=0)

        cropped = resize_frame(snap[positive_index], (299, 299))
        depth_cropped = resize_frame(snap_depth[positive_index], (299, 299))
        positive_frame = np.concatenate([cropped, _smooth(depth_cropped[None, 0])], axis=0)

        cropped = resize_frame(snap[negative_index], (299, 299))
        depth_cropped = resize_frame(snap_depth[negative_index], (299, 299))
        depth_cropped = np.transpose(depth_cropped, [2, 0, 1])
        negative_frame = np.concatenate([cropped, _smooth(depth_cropped[None, 0])], axis=0)

        # except:
        #     set_trace()
        #     print(self.video_index)
        #     print(len(snap))
        #     print(len(snap_depth))


        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame), torch.zeros(anchor_frame.shape), torch.zeros(anchor_frame.shape),
            torch.zeros(anchor_frame.shape))


    def build_set(self):
        # build image triplet item
        # build caption item
        triplets = []
        triplets = torch.Tensor(self.sample_size, 3, 4, *self.frame_size)
        triplets_features = torch.zeros(triplets.size())
        snap = self.get_extracted_video(self.video_index)
        depth_snap = self.get_extracted_depth_video(self.video_index)
        results_snap = self.get_extracted_rcnn_results(self.video_index)

        for i in range(0, self.sample_size):

            anchor_frame, positive_frame, negative_frame, \
                anchor_feature, positive_feature, \
                negative_feature = self.sample_triplet(snap, depth_snap)
            triplets[i, 0, :, :, :] = anchor_frame
            triplets[i, 1, :, :, :] = positive_frame
            triplets[i, 2, :, :, :] = negative_frame
            triplets_features[i, 0, :, :, :] = anchor_feature
            triplets_features[i, 1, :, :, :] = positive_feature
            triplets_features[i, 2, :, :, :] = negative_feature


        print(self.video_index)   
        # active_label = self.vocab(label[-4])
        # passive_label = self.vocab(label[-2])
        # Convert caption (string) to word ids.
        self.video_index = (self.video_index + 1) % self.video_count
        return TensorDataset(triplets, triplets_features)


    def _read_label_dir(self, label_directory):
        self._label_directory = label_directory
        filenames = ls_txt(label_directory)
        self.label_paths = [os.path.join(self._label_directory, f) for f in filenames]
        
    def get_caption(self, index):
        caption = read_caption(self.caption_paths[index])
        return caption

    def get_extracted_video(self, index):
        return read_extracted_video(self.video_paths[index], self.frame_size)

    def get_extracted_depth_video(self, index):
        return read_extracted_video(self.depth_video_paths[index], self.frame_size)

    def get_extracted_rcnn_results(self, index):
        return read_extracted_rcnn_results(self.video_paths[index], self.frame_size)

    def concat_frames(self, images):
        for i, img in enumerate(images):
            if i == 0:
                concat_img = img
                continue
            else:
                concat_img = np.concatenate([concat_img, img], axis=1)
        plt.imshow(concat_img)
        plt.show()
        return concat_img

    def sample_anchor_frame_index(self, video):
        arange = np.arange(0, len(video))
        return np.random.choice(arange)

    def sample_positive_frame_index(self, anchor_index, video):
        video_length = len(video)
        lower_bound = max(0, anchor_index - self.positive_frame_margin)
        range1 = np.arange(lower_bound, anchor_index)
        upper_bound = min(video_length - 1, anchor_index + self.positive_frame_margin)
        range2 = np.arange(anchor_index + 1, upper_bound)
        return np.random.choice(np.concatenate([range1, range2]))

    def negative_frame_indices(self, anchor_index, video):
        video_length = len(video)
        lower_bound = 0
        upper_bound = max(0, anchor_index - self.negative_frame_margin)
        range1 = np.arange(lower_bound, upper_bound)
        lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
        upper_bound = video_length 
        range2 = np.arange(lower_bound, upper_bound)
        return np.concatenate([range1, range2])

    def sample_negative_frame_index(self, anchor_index, video):
        return np.random.choice(self.negative_frame_indices(anchor_index, video))

class TwoFrameInverseBuilder(object):
    # Assumes that all videos/views are trained as single training examples, i.e. every view is its own sample
    def __init__(self, view, video_directory, image_size, cli_args, sample_size=500):
        self.frame_size = image_size
        self.view = view
        self._read_video_dir(video_directory)
        self._count_frames()
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.negative_frame_margin = 10 
        self.cli_args = cli_args
        self.sample_size = sample_size
        self.video_index = 0
        self.skipped_frames = 4

    def _read_video_dir(self, video_directory):
        self._video_directory = video_directory
        filenames = ls(video_directory)
        self.video_paths = [os.path.join(self._video_directory, f) for f in filenames if int(f.split('_')[0]) ==  self.view]
        # for path in self.video_paths:
        #     print(path)
        self.video_count = len(self.video_paths)

    def _count_frames(self):
        frame_lengths = np.array([len(imageio.read(p)) for p in self.video_paths])
        self.frame_lengths = frame_lengths - OFFSET
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames
   
    @functools.lru_cache(maxsize=1)
    def get_delta_euler(self, index, type='ee'):
        # pose is (T, 4)
        seqname = self.video_paths[index].split('/')[-1].split('_')[0]
        filepath = '/'.join(self.video_paths[index].split('/')[:-1]) + '/'  + seqname + '_' + type + '.npy'
        pose = np.load(filepath)[:, -4:]
        delta_euler = np.zeros((len(pose), 3))
        for i in range(len(pose) - 1 - self.skipped_frames):
            pose[:, [0,3]] = pose[:, [3, 0]]
            q1 = Quaternion(pose[i])
            q2 = Quaternion(pose[i + 1 + self.skipped_frames])
            delta_q = q2*q1.inverse
            delta_euler[i] = rotationMatrixToEulerAngles(delta_q.rotation_matrix)
        return delta_euler

    @functools.lru_cache(maxsize=1)
    def get_video(self, index):
        return read_video(self.video_paths[index], self.frame_size)

    def sample(self, snaps, delta_euler):
        anchor_frames = np.zeros((1 + 1, 3, 299, 299))
        anchor_index = self.sample_anchor_frame_index()
        anchor_frames[0] = snaps[anchor_index]
        anchor_delta_euler = delta_euler[anchor_index]

        anchor_frames[1] = snaps[anchor_index + 1 + self.skipped_frames]
        return (torch.Tensor(anchor_frames), torch.Tensor(anchor_delta_euler))
            

    def build_set(self):
        frames = torch.Tensor(self.sample_size, 1 + 1, 3, *self.frame_size)
        delta_eulers = torch.Tensor(self.sample_size, 3)
        #print(self.video_paths[self.video_index])
        snap = self.get_video(self.video_index)
        delta_euler = self.get_delta_euler(self.video_index)
        for i in range(0, self.sample_size):
            anchor_frame, anchor_delta_euler = self.sample(snap, delta_euler)
            frames[i, :, :, :] = anchor_frame
            delta_eulers[i, :] = anchor_delta_euler
        self.video_index = (self.video_index + 1) % self.video_count
        # Second argument is labels. Not used.
        return TensorDataset(frames, delta_eulers)

    def sample_anchor_frame_index(self):
        arange = np.arange(0 , self.frame_lengths[self.video_index] - 1 - self.skipped_frames)
        return np.random.choice(arange)



class SingleViewMultiFrameTripletBuilder(object):
    def __init__(self, view, n_prev_frames, video_directory, image_size, cli_args, sample_size=500):
        self.frame_size = image_size
        self.view = view
        self.n_prev_frames = n_prev_frames
        self._read_video_dir(video_directory)
        self._count_frames()
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.negative_frame_margin = 10 
        self.cli_args = cli_args
        self.sample_size = sample_size
        self.video_index = 0

    def _read_video_dir(self, video_directory):
        self._video_directory = video_directory
        filenames = ls(video_directory)
        self.video_paths = [os.path.join(self._video_directory, f) for f in filenames]
        # for path in self.video_paths:
        #     print(path)
        self.video_count = len(self.video_paths)

    def _count_frames(self):
        frame_lengths = np.array([len(imageio.read(p)) for p in self.video_paths])
        self.frame_lengths = frame_lengths - OFFSET
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    @functools.lru_cache(maxsize=1)
    def get_video(self, index):
        return read_video(self.video_paths[index], self.frame_size)

    def sample_triplet(self, snaps):
        loaded_sample = False
        while not loaded_sample:

            try:
                anchor_index = self.sample_anchor_frame_index()
                positive_index = anchor_index
                negative_index = self.sample_negative_frame_index(anchor_index)
                loaded_sample = True
            except:
                pass
                # print("Error loading video - sequence index: ", self.sequence_index)
                # print("video lengths: ", [len(snaps[i]) for i in range(0, len(snaps))])
                # print("Maybe margin too high")
        # random sample anchor view,and positive view

        anchor_frame = np.zeros((self.n_prev_frames + 1, 3, 299, 299))
        positive_frame = np.zeros((self.n_prev_frames + 1, 3, 299, 299))
        negative_frame = np.zeros((self.n_prev_frames + 1, 3, 299, 299))
        anchor_frame[0] = snaps[anchor_index]
        positive_frame[0] = snaps[positive_index]
        negative_frame[0] = snaps[negative_index]
        
        for ii in range(1, self.n_prev_frames+1):
            
            anchor_frame[ii] = snaps[anchor_index-ii]
            positive_frame[ii] = snaps[positive_index-ii]
            negative_frame[ii] = snaps[negative_index-ii]

        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame))

    def build_set(self):
        triplets = []
        triplets = torch.Tensor(self.sample_size, self.n_prev_frames + 1, 3, 3, *self.frame_size)
        #print(self.video_paths[self.video_index])
        for i in range(0, self.sample_size):
            snaps = self.get_video(self.video_index)
            anchor_frame, positive_frame, negative_frame = self.sample_triplet(snaps)
            triplets[i, :, 0, :, :, :] = anchor_frame
            triplets[i, :, 1, :, :, :] = positive_frame
            triplets[i, :, 2, :, :, :] = negative_frame
        self.video_index = (self.video_index + 1) % self.video_count
        # Second argument is labels. Not used.
        return TensorDataset(triplets, torch.zeros(triplets.size()[0]))

    def sample_anchor_frame_index(self):
        arange = np.arange(0 + self.n_prev_frames, self.frame_lengths[self.video_index])
        return np.random.choice(arange)

    # def sample_positive_frame_index(self, anchor_index):
    #     upper_bound = min(self.frame_lengths[self.sequence_index * self.n_views + 1] - 1, anchor_index)
    #     return upper_bound # in case video has less frames than anchor video

    def negative_frame_indices(self, anchor_index):
        video_length = self.frame_lengths[self.video_index]
        lower_bound = self.n_prev_frames
        upper_bound = max(0, anchor_index - self.negative_frame_margin)
        range1 = np.arange(lower_bound, upper_bound)
        lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
        upper_bound = video_length
        range2 = np.arange(lower_bound, upper_bound)
        return np.concatenate([range1, range2])

    def sample_negative_frame_index(self, anchor_index):
        return np.random.choice(self.negative_frame_indices(anchor_index))

class MultiViewMultiFrameTripletBuilder(object):
    def __init__(self, n_views, n_prev_frames, video_directory, image_size, cli_args, sample_size=500):
        self.frame_size = image_size
        self.n_views = n_views
        self.n_prev_frames = n_prev_frames
        self._read_video_dir(video_directory)

        self._count_frames()
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.negative_frame_margin = 5 
        self.sequence_index = 0
        self.cli_args = cli_args
        self.sample_size = sample_size

    def _read_video_dir(self, video_directory):
        self._video_directory = video_directory
        filenames = ls(video_directory)
        self.video_paths = [os.path.join(self._video_directory, f) for f in filenames]
        # for path in self.video_paths:
        #     print(path)
        self.sequence_count = int(len(self.video_paths) / self.n_views)

    def _count_frames(self):
        frame_lengths = np.array([len(imageio.read(p)) for p in self.video_paths])
        self.frame_lengths = frame_lengths - OFFSET
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    @functools.lru_cache(maxsize=1)
    def get_videos(self, index):
        views = []
        for i in range(self.n_views):
            views.append(read_video(self.video_paths[index + i], self.frame_size))
        return views

    def sample_triplet(self, snaps):
        loaded_sample = False
        while not loaded_sample:

            try:
                anchor_index = self.sample_anchor_frame_index()
                positive_index = anchor_index
                negative_index = self.sample_negative_frame_index(anchor_index)
                loaded_sample = True
            except:
                pass
                # print("Error loading video - sequence index: ", self.sequence_index)
                # print("video lengths: ", [len(snaps[i]) for i in range(0, len(snaps))])
                # print("Maybe margin too high")
        # random sample anchor view,and positive view
        view_set = set(range(self.n_views))
        anchor_view = np.random.choice(np.array(list(view_set)))
        view_set.remove(anchor_view)
        positive_view = np.random.choice(np.array(list(view_set)))
        negative_view = anchor_view # negative example comes from same view INQUIRE TODO

        anchor_frame = np.zeros((self.n_prev_frames + 1, 3, 299, 299))
        positive_frame = np.zeros((self.n_prev_frames + 1, 3, 299, 299))
        negative_frame = np.zeros((self.n_prev_frames + 1, 3, 299, 299))
        anchor_frame[0] = snaps[anchor_view][anchor_index]
        positive_frame[0] = snaps[positive_view][positive_index]
        negative_frame[0] = snaps[negative_view][negative_index]
        
        for ii in range(1, self.n_prev_frames+1):
              
            anchor_frame[ii] = snaps[anchor_view][anchor_index-ii]
            positive_frame[ii] = snaps[positive_view][positive_index-ii]
            negative_frame[ii] = snaps[negative_view][negative_index-ii]


        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame))

    def build_set(self):
        triplets = []
        triplets = torch.Tensor(self.sample_size, self.n_prev_frames + 1, 3, 3, *self.frame_size)
        for i in range(0, self.sample_size):
            snaps = self.get_videos(self.sequence_index * self.n_views)
            anchor_frame, positive_frame, negative_frame = self.sample_triplet(snaps)
            triplets[i, :, 0, :, :, :] = anchor_frame
            triplets[i, :, 1, :, :, :] = positive_frame
            triplets[i, :, 2, :, :, :] = negative_frame
        self.sequence_index = (self.sequence_index + 1) % self.sequence_count
        # Second argument is labels. Not used.
        return TensorDataset(triplets, torch.zeros(triplets.size()[0]))

    def sample_anchor_frame_index(self):
        arange = np.arange(0 + self.n_prev_frames, self.frame_lengths[self.sequence_index * self.n_views])
        return np.random.choice(arange)

    # def sample_positive_frame_index(self, anchor_index):
    #     upper_bound = min(self.frame_lengths[self.sequence_index * self.n_views + 1] - 1, anchor_index)
    #     return upper_bound # in case video has less frames than anchor video

    def negative_frame_indices(self, anchor_index):
        video_length = self.frame_lengths[self.sequence_index * self.n_views]
        lower_bound = self.n_prev_frames
        upper_bound = max(0, anchor_index - self.negative_frame_margin)
        range1 = np.arange(lower_bound, upper_bound)
        lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
        upper_bound = video_length
        range2 = np.arange(lower_bound, upper_bound)
        return np.concatenate([range1, range2])

    def sample_negative_frame_index(self, anchor_index):
        return np.random.choice(self.negative_frame_indices(anchor_index))


class MultiViewDepthTripletBuilder(MultiViewTripletBuilder):
    def __init__(self, n_views, video_directory, depth_video_directory, image_size, cli_args, sample_size=500):
        super(MultiViewDepthTripletBuilder, self).__init__(n_views, video_directory, image_size, cli_args, sample_size)
        self._read_depth_video_dir(depth_video_directory)

    def _read_depth_video_dir(self, depth_video_directory):
        self._depth_video_directory = depth_video_directory
        filenames = ls(depth_video_directory)
        self.depth_video_paths = [os.path.join(self._depth_video_directory, f) for f in filenames]
        self.depth_video_count = len(self.depth_video_paths)


    def sample_triplet(self, snaps, depth_snaps):
        loaded_sample = False
        while not loaded_sample:

            try:
                anchor_index = self.sample_anchor_frame_index()
                positive_index = anchor_index
                negative_index = self.sample_negative_frame_index(anchor_index)
                loaded_sample = True
            except:
                pass
                # print("Error loading video - sequence index: ", self.sequence_index)
                # print("video lengths: ", [len(snaps[i]) for i in range(0, len(snaps))])
                # print("Maybe margin too high")
        # random sample anchor view,and positive view
        view_set = set(range(self.n_views))
        anchor_view = np.random.choice(np.array(list(view_set)))
        view_set.remove(anchor_view)
        positive_view = np.random.choice(np.array(list(view_set)))
        negative_view = anchor_view # negative example comes from same view INQUIRE TODO
        anchor_frame = np.concatenate([snaps[anchor_view][anchor_index], depth_snaps[anchor_view][anchor_index][None, 0]], axis=0)
        positive_frame = np.concatenate([snaps[positive_view][positive_index], depth_snaps[positive_view][positive_index][None, 0]], axis=0)
        negative_frame = np.concatenate([snaps[negative_view][negative_index], depth_snaps[negative_view][negative_index][None, 0]], axis=0)

        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame))

    def build_set(self):
        triplets = []
        triplets = torch.Tensor(self.sample_size, 3, 4, *self.frame_size)
        for i in range(0, self.sample_size):
            snaps = self.get_videos(self.sequence_index * self.n_views)
            depth_snaps = self.get_depth_videos(self.sequence_index * self.n_views)

            anchor_frame, positive_frame, negative_frame = self.sample_triplet(snaps, depth_snaps)
            triplets[i, 0, :, :, :] = anchor_frame
            triplets[i, 1, :, :, :] = positive_frame
            triplets[i, 2, :, :, :] = negative_frame
        print(self.sequence_index)

        self.sequence_index = (self.sequence_index + 1) % self.sequence_count
        # Second argument is labels. Not used.
        return TensorDataset(triplets, torch.zeros(triplets.size()[0]))

    def get_depth_videos(self, index):
        views = []
        for i in range(self.n_views):
            views.append(read_video(self.depth_video_paths[index + i], self.frame_size))
        return views 

class SingleViewPoseBuilder(object):
    def __init__(self, view, video_directory, image_size, cli_args, toRot=False, sample_size=500):
        self.frame_size = image_size
        self.view = view
        self._read_video_dir(video_directory)
        self._count_frames()
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.positive_frame_margin = 2
        self.negative_frame_margin = 4
        self.video_index = 0
        self.cli_args = cli_args
        self.sample_size = sample_size
        self.toRot = toRot

    def _read_video_dir(self, video_directory):
        self._video_directory = video_directory
        filenames = ls_view(video_directory, 2)
        self.video_paths = [os.path.join(self._video_directory, f) for f in filenames if f.endswith('.mp4')]
        self.video_count = len(self.video_paths)
        

    def _read_extracted_video_dir(self, video_directory):
        self._video_directory = video_directory
        filenames = ls_extracted(video_directory)
        # self.video_paths = [os.path.join(self._video_directory, f.split('.mp4')[0]) for f in filenames]
        self.video_paths = [os.path.join(self._video_directory, f) for f in filenames if not f.endswith('.mp4') and 'debug' not in f]
        self.video_count = len(self.video_paths)

    def _count_frames(self):
        frame_lengths = np.array([len(imageio.read(p)) for p in self.video_paths])
        self.frame_lengths = frame_lengths
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    def _count_extracted_frames(self):
        frame_lengths = np.array([len(os.listdir(p)) for p in self.video_paths if p.endswith('.jpg')])
        self.frame_lengths = frame_lengths
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    @functools.lru_cache(maxsize=1)
    def get_video(self, index):
        return read_video(self.video_paths[index], self.frame_size)
    
    @functools.lru_cache(maxsize=1)
    def get_pose(self, index):
        # PyQuaternion: WXYZ, pybullet> XYZW
        try:
            pose = np.load(self.video_paths[index].split('view')[0]+'obj.npy')[:, -4:]
            return pose
        except:
            try:
                pose = np.load(self.video_paths[index].split('.mp4')[0]+'.npy')[:, -4:]
                return pose
            except:
                pose = np.load(self.video_paths[index].split('_cropped.mp4')[0]+'.npy')[:, -4:]
                return pose

    def sample(self, snap, pose):
        anchor_index = self.sample_anchor_frame_index()
        anchor_frame = snap[anchor_index]
        anchor_frame = np.transpose(np.asarray(content_transform(np.uint8(np.transpose(255*anchor_frame, [1,2,0])))) / 255.0, [2,0,1])
        anchor_pose_qt = pose[anchor_index]
        if self.toRot:
            anchor_pose_qt = Quaternion(anchor_pose_qt[[-1, 0, 1, 2]])
            anchor_pose = anchor_pose_qt.rotation_matrix 
        else:
            anchor_pose = anchor_pose_qt
        return (torch.Tensor(anchor_frame), torch.Tensor(anchor_pose))

    def build_set(self):
        triplets = []
        frames = torch.Tensor(self.sample_size, 3, *self.frame_size)
        if self.toRot:
            poses = torch.Tensor(self.sample_size, 3, 3)
        else:
            poses = torch.Tensor(self.sample_size, 4)
        # print("Create triplets from {}".format(self.video_paths[self.video_index]))        
        snap = self.get_video(self.video_index)
        pose = self.get_pose(self.video_index)
        
        for i in range(0, self.sample_size):
            anchor_frame, anchor_pose = self.sample(snap, pose)
            frames[i, :, :, :] = anchor_frame
            poses[i, :] = anchor_pose
        self.video_index = (self.video_index + 1) % self.video_count
        # Second argument is labels. Not used.
        return TensorDataset(frames, poses)

    def sample_anchor_frame_index(self):
        arange = np.arange(0, self.frame_lengths[self.video_index])
        return np.random.choice(arange)

    def sample_positive_frame_index(self, anchor_index):
        lower_bound = max(0, anchor_index - self.positive_frame_margin)
        range1 = np.arange(lower_bound, anchor_index)
        upper_bound = min(self.frame_lengths[self.video_index] - 1, anchor_index + self.positive_frame_margin)
        range2 = np.arange(anchor_index + 1, upper_bound)
        return np.random.choice(np.concatenate([range1, range2]))

    def negative_frame_indices(self, anchor_index):
        video_length = self.frame_lengths[self.video_index]
        lower_bound = 0
        upper_bound = max(0, anchor_index - self.negative_frame_margin)
        range1 = np.arange(lower_bound, upper_bound)
        lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
        upper_bound = video_length 
        range2 = np.arange(lower_bound, upper_bound)
        return np.concatenate([range1, range2])

    def sample_negative_frame_index(self, anchor_index):
        return np.random.choice(self.negative_frame_indices(anchor_index))


