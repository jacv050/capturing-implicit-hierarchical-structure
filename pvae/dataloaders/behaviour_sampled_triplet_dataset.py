import torch
from torch.utils import data
import pickle
import numpy as np
from skimage.transform import rescale
from os import listdir
from os.path import join
from os.path import isfile
import sys
import glob
import torchvision
from torchvision import transforms
from torchvision.datasets.video_utils import VideoClips
from pvae.dataloaders import utils
import pickle
from math import ceil
import multiprocessing.dummy as mp
import threading
import random
import gc
#import open3d as o3d

mutex = threading.Lock()
#mutexv = threading.Lock()

class BehaviourSampledTripletDataset(data.Dataset):
    def __init__(self, width, height, depth, no_background):

        self.width = width
        self.height = height
        self.depth = depth

        self.threads = 10

        #gray scale tensor and resize
        sequential = torch.nn.Sequential(transforms.Grayscale(), transforms.Resize((200, 200))) #, ]) #TODO parameterize
        self.transforms = torch.jit.script(sequential)

        #all_inputs = []
        #List of videos
        data_dir = '../dataset/p1/test/'

        metadata_location = "../dataset/p1/metadata.pt"
        _precomputed_metadata = None
        ignore_last_metadata = False
        if(isfile(metadata_location) and not ignore_last_metadata):
            _precomputed_metadata = torch.load(metadata_location)

        self.input_toms = glob.glob(data_dir + '*.mp4')
        self.video_clips = VideoClips(
            self.input_toms,
            20, #frames_per_clip/clip_length_in_frames
            20, #step_between_clips/frames_between_clips
            4, #frame_rate
            _precomputed_metadata, #_precomputed_metadata
            num_workers = 10,
            _video_width = 0,
            _video_height = 0,
            _video_min_dimension = 0,
            _audio_samples=0,
        )
        self.mutex_list = [threading.Lock() for i in range(self.video_clips.num_videos())]
        #self.video_reading = [False for i in range(self.video_clips.num_videos())]

        print(self.video_clips.num_clips())
        if(_precomputed_metadata is None):
            torch.save(self.video_clips.metadata, metadata_location)

        folder = 'patched'
        #Files after video load and triplet relation
        self.index_mapping = 'preprocessed/{}/index_video_{}.pt'.format(folder, '{}')
        #Files befores triplet after video load
        self.l_mapping = 'preprocessed/{}/l_video_{}.pt'.format(folder, '{}')
        self.no_background = no_background
        if(len(glob.glob(self.index_mapping.format('*'))) == 0):
            self.preprocess_video_patches(self.input_toms)

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def patch_video_3D(self, l, ind_input):
        for i in range(2, 48, 1):
            for j in range(2, 48, 1):
                for k in range(2, 48, 1):

                    sampled_patch_size = np.random.randint(5, 35)
                    sampled_patch_margin = int(sampled_patch_size / 2)

                    mins = []
                    origins = [i, j, k]
                    for origin in origins:
                        if origin + sampled_patch_margin >= 50:
                            origin_sampled_patch_size = 49 - origin
                            mins.append(origin_sampled_patch_size * 2)

                        if origin - sampled_patch_margin < 0:
                            origin_sampled_patch_size = origin - 0
                            mins.append(origin_sampled_patch_size * 2)
                    if len(mins) != 0:
                        sampled_patch_size = np.min(mins)
                        sampled_patch_margin = int(sampled_patch_size / 2)

                    if self.no_background:
                        #ind_input = all_inputs[l]
                        #ind_input = self.l_mapping_ind[l] TODO
                        parent_patch = ind_input[i-sampled_patch_margin:i+sampled_patch_margin+1,
                                              j-sampled_patch_margin:j+sampled_patch_margin+1,
                                              k-sampled_patch_margin:k+sampled_patch_margin+1]

                        count = np.count_nonzero(parent_patch == parent_patch[0, 0, 0])

                        # append if not background
                        if count != (sampled_patch_size * sampled_patch_size * sampled_patch_size):

                            if sampled_patch_size - 1 <= 3:
                                child_patch_size = 3
                            else:
                                child_patch_size = np.random.randint(3, min(15, sampled_patch_size - 1))
                            child_patch_margin = int(child_patch_size / 2)

                            potential_origins = set(range(child_patch_margin, 50-child_patch_margin))
                            parent_i_points = set(range(i-sampled_patch_margin, i+sampled_patch_margin+1))
                            parent_j_points = set(range(j-sampled_patch_margin, j+sampled_patch_margin+1))
                            parent_k_points = set(range(k-sampled_patch_margin, k+sampled_patch_margin+1))

                            potential_i_origins = list(potential_origins - parent_i_points)
                            negative_i = np.random.choice(potential_i_origins, 1)[0]

                            potential_j_origins = list(potential_origins - parent_j_points)
                            negative_j = np.random.choice(potential_j_origins, 1)[0]

                            potential_k_origins = list(potential_origins - parent_k_points)
                            negative_k = np.random.choice(potential_k_origins, 1)[0]

                            return ((l, i, j, k, sampled_patch_size),
                                              (l, i, j, k, child_patch_size),
                                              (l, negative_i, negative_j, negative_k, child_patch_size))
                        else:
                            return None

                    else:
                        child_patch_size = np.random.randint(3, sampled_patch_size)
                        child_patch_margin = int(child_patch_size / 2)

                        potential_origins = set(range(child_patch_margin, 50-child_patch_margin))
                        parent_i_points = set(range(i-sampled_patch_margin, i+sampled_patch_margin+1))
                        parent_j_points = set(range(j-sampled_patch_margin, j+sampled_patch_margin+1))
                        parent_k_points = set(range(k-sampled_patch_margin, k+sampled_patch_margin+1))

                        potential_i_origins = list(potential_origins - parent_i_points)
                        negative_i = np.random.choice(potential_i_origins, 1)[0]

                        potential_j_origins = list(potential_origins - parent_j_points)
                        negative_j = np.random.choice(potential_j_origins, 1)[0]

                        potential_k_origins = list(potential_origins - parent_k_points)
                        negative_k = np.random.choice(potential_k_origins, 1)[0]

                        return ((l, i, j, k, sampled_patch_size),
                                          (l, i, j, k, child_patch_size),
                                          (l, negative_i, negative_j, negative_k, child_patch_size))


    def preprocess_thread(self, l):
        #print('Preprocessing: {}'.format(l))
        #ind_input = self.read_video(all_inputs[l])
        #If video is cached we load from preprocessed #TODO Define file with video params
        #ind_input = self.read_video(l)
        video_idx, clip_idx = self.video_clips.get_clip_location(l)
        #self.mutex_list[video_idx].acquire()
        video, _, _ , _ = self.video_clips.get_clip(l)
        #ind_input = torch.zeros((50,50,50))
        #self.mutex_list[video_idx].release()
        ind_input = utils.read_video_shape(video, 50.0, 50.0, 50.0)

        torch.save(ind_input, self.l_mapping.format(l))
        del(video)

        patch = self.patch_video_3D(l, ind_input)
        del(ind_input)
        if(patch is not None):
            #path = all_inputs[l]
            #TODO Define file with video params
            mutex.acquire()
            mindex = self.index
            self.index += 1
            mutex.release()
            torch.save(patch, self.index_mapping.format(mindex))
        del(patch)
        gc.collect()

    def preprocess_video_patches(self, all_inputs):
        self.index = 0
        #for l in range(len(all_inputs)): #self.video_clips.num_videos()
        p = mp.Pool(self.threads)
        lst = list(range(self.video_clips.num_clips()))
        random.shuffle(lst)
        for i, _ in enumerate(p.imap_unordered(self.preprocess_thread, lst, 1)):
            sys.stderr.write('\rdone {0:%}\n'.format(i/self.video_clips.num_clips()))
        print("")
        p.close()
        p.join()
        gc.collect()

    def __len__(self):
        return len(glob.glob(self.index_mapping.format('*')))

    def __getitem__(self, index):
        item = torch.load(self.index_mapping.format(index))
        ((l, i, j, k, sampled_patch_size),
         (l, i, j, k, child_patch_size),
         (l, negative_i, negative_j, negative_k, child_patch_size)) = item #self.index_mapping[index]

        sampled_patch_margin = int(sampled_patch_size / 2)
        child_patch_margin = int(child_patch_size / 2)

        #ind_input = self.input_toms[l]
        ind_input = torch.load(self.l_mapping.format(l))

        #shape -> (5, 5, 5)
        parent_patch = ind_input[i-sampled_patch_margin:i+sampled_patch_margin+1,
                                 j-sampled_patch_margin:j+sampled_patch_margin+1,
                                 k-sampled_patch_margin:k+sampled_patch_margin+1]
        
        positive_child = ind_input[i-child_patch_margin:i+child_patch_margin+1,
                                   j-child_patch_margin:j+child_patch_margin+1,
                                   k-child_patch_margin:k+child_patch_margin+1]
        
        
        negative_child = ind_input[negative_i-child_patch_margin:negative_i+child_patch_margin+1,
                                   negative_j-child_patch_margin:negative_j+child_patch_margin+1,
                                   negative_k-child_patch_margin:negative_k+child_patch_margin+1]

        print("Original parent shape: {}".format(parent_patch.shape))
        print("Sampled patch margin: {}".format(sampled_patch_margin))
        parent_patch = rescale(parent_patch, 10.0/(sampled_patch_margin * 2 + 1), anti_aliasing=False)
        parent_patch = np.expand_dims(parent_patch, 0)

        #Rescale to [10, 10, 10]
        positive_child = rescale(positive_child, 10.0/(child_patch_margin * 2 + 1), anti_aliasing=False)
        positive_child = np.expand_dims(positive_child, 0)
        
        negative_child = rescale(negative_child, 10.0/(child_patch_margin * 2 + 1), anti_aliasing=False)
        negative_child = np.expand_dims(negative_child, 0)

        print(parent_patch.shape)
        print(positive_child.shape)
        print(negative_child.shape)
        sys.exit(-1)
        return parent_patch, positive_child, negative_child, (1.0 - ((sampled_patch_margin * 2 + 1)/35.0))

    #jx, jz, jz size of jump in each dimension
    def conv_3dmean(self, jx, jy, jz, input3d):
        output = torch.zeros(50,50,50)
        for xi in range(50):
            for yi in range(50):
                for zi in range(50):
                    #mean of subsquare
                    #cell = input3d[xi*jx:jx,yi*jy:jy,zi*jz:jz].mean()
                    z = zi*jz
                    x = xi*jx
                    y = yi*jy
                    m = input3d[z:z+jz, :, x:x+jx, y:y+jy]
                    if(m.shape[0] != 0 and m.shape[2] != 0 and m.shape[3] != 0):
                        cell = m.mean()
                        output[xi,yi,zi] = cell
        return output

    #Video to tensor
    def read_video(self, l):
        #video, audio, info = torchvision.io.read_video(path, start_pts=0, end_pts=None, pts_unit='sec')
        video, audio, info, video_idx = self.video_clips.get_clip(l)
        #resize to box
        #video_squared = self.transforms(video)
        #[N x W x H x C] -> [N x C x W x H]
        video = video.permute((0, 3, 1, 2))
        jump_x = ceil(video.shape[2]/50.0)
        jump_y = ceil(video.shape[3]/50.0)
        jump_z = ceil(video.shape[0]/50.0)
        #convolved_video = []
        #print(v.shape)
        video_t = self.transforms(video)
        convolved_video = self.conv_3dmean(jump_x, jump_y, jump_z, video_t.float())
        del video
        del video_t
        gc.collect()
        #sys.exit()
        #Apply transforms
        return convolved_video

