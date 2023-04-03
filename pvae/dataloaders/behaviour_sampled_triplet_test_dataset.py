import torch
from torch.utils import data
import pickle
import numpy as np
from skimage.transform import rescale
from os import listdir
from os.path import join
from os.path import isfile
import glob
from torchvision import transforms
from torchvision.datasets.video_utils import VideoClips
from pvae.dataloaders import utils
import sys
import threading
import random
import multiprocessing.dummy as mp
import gc

mutex = threading.Lock()

class BehaviourSampledTripletTestDataset(data.Dataset):
    def __init__(self, width, height, depth):

        self.width = width
        self.height = height
        self.depth = depth

        self.vx = 50
        self.vy = 50
        self.vz = 50
        self.shapes = 3

        #Gray scale tensor and resize
        sequential = torch.nn.Sequential(transforms.Grayscale(), transforms.Resize((200, 200)))
        self.transforms = torch.jit.script(sequential)

        self.threads = 10

        data_dir = '../dataset/p1/test/'

        metadata_location = '../dataset/p1/metadata.pt'
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
            num_workers = 0,
            _video_width = 0,
            _video_height = 0,
            _video_min_dimension = 0,
            _audio_samples=0,
        )
        if(_precomputed_metadata is None):
            torch.save(self.video_clips.metadata, metadata_location)

        folder = 'patched'
        self.l_mapping = 'preprocessed/{}/l_test_video_{}.pt'.format(folder, '{}')
        self.index_mapping = 'preprocessed/{}/index_test_video.pt'.format(folder)
        if(not isfile(self.index_mapping)):
            self.preprocess()
        else:
            self.mapping = torch.load(self.index_mapping)

    def preprocess_thread(self, l):

        if(isfile(self.l_mapping.format(l))):
            return
        video_idx, clip_idx = self.video_clips.get_clip_location(l)
        video, _, _, _ = self.video_clips.get_clip(l)
        toy = utils.read_video_shape(video, 50.0, 50.0, 50.0, self.transforms)
        del(video)

        padded_toy = torch.zeros((64, 64, 64))
        padded_toy[8:58, 8:58, 8:58] = toy
        torch.save(padded_toy, self.l_mapping.format(l))
        del(padded_toy)

        gc.collect()

    def generate_index_map(self, samples):
        size = self.vx*self.vy*self.vz*self.shapes
        map = torch.empty((size, 5), dtype=torch.int)
        index = 0
        # 5 x 5
        #for l in range(toys):
        for i in range(6, 56, 1):
            for j in range(6, 56, 1):
                for k in range(6, 56, 1):
                    #item = (l, i, j, k, 5)
                    map[index, 1:] = torch.tensor([i, j, k, 5])
                    #self.mapping[index] = item
                    index += 1

        # 10 x 10
        #for l in range(toys):
        for i in range(4, 54, 1):
            for j in range(4, 54, 1):
                for k in range(4, 54, 1):
                    #item = (l, i, j, k, 10)
                    map[index, 1:] = torch.tensor([i, j, k, 10])
                    #self.mapping[index] = item
                    index += 1

        # 15 x 15
        #for l in range(toys):
        for i in range(0, 50, 1):
            for j in range(0, 50, 1):
                for k in range(0, 50, 1):
                    #item = (l, i, j, k, 15)
                    map[index, 1:] = torch.tensor([i, j, k, 15])
                    #self.mapping[index] = item
                    index += 1

        map = map.repeat(samples,1)
        l = torch.tensor(range(samples), dtype=torch.int)
        l = l.repeat_interleave(size)
        map[:, 0] = l
        self.mapping = map
        torch.save(self.mapping, self.index_mapping)

    def preprocess(self):
        samples = self.video_clips.num_clips()
        self.generate_index_map(samples)
        #size = self.vx*self.vy*self.vz*self.shapes*samples
        #self.mapping = [None for i in range(size)]

        self.index = 0
        #self.mapping = {}
        p = mp.Pool(self.threads)
        lst = list(range(samples))
        random.shuffle(lst)
        for i, _ in enumerate(p.imap_unordered(self.preprocess_thread, lst, 1)):
            sys.stderr.write('\rdone {0:%}'.format(i/self.video_clips.num_clips()))
        print("")
        p.close()
        p.join()

    def __len__(self):
        return self.mapping.shape[0]
        #return len(glob.glob(self.index_mapping.format('*')))

    def __getitem__(self, index):
        item = self.mapping[index]
        l, i, j, k, patch_size = item

        original_tom = torch.load(self.l_mapping.format(l))

        if patch_size == 5:
            original_tomogram = original_tom[i:i+5, j:j+5, k:k+5]
            original_tomogram = rescale(original_tomogram, 2, anti_aliasing=False)
            original_tomogram = np.expand_dims(original_tomogram, 0)

        if patch_size == 10:
            original_tomogram = original_tom[i:i+10, j:j+10, k:k+10]
            original_tomogram = np.expand_dims(original_tomogram, 0)

        if patch_size == 15:
            original_tomogram = original_tom[i:i+15, j:j+15, k:k+15]
            original_tomogram = rescale(original_tomogram, 2/3, anti_aliasing=False)
            original_tomogram = np.expand_dims(original_tomogram, 0)

        return original_tomogram, patch_size
