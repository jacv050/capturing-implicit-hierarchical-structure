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
import os
import tqdm

mutex = threading.Lock()

class BehaviourVideoTripletTestDataset(data.Dataset):
    def __init__(self, width, height, depth, no_padding=True, num_images=1):
        self.no_padding = True#no_padding
        self.width = width
        self.height = height
        self.depth = depth
        self.num_images = num_images

        self.vx = 50.0
        self.vy = 50.0
        self.vz = 50.0
        self.shapes = 3

        #Gray scale tensor and resize
        #sequential = torch.nn.Sequential(transforms.Grayscale(), transforms.Resize((200, 200)))
        #self.transforms = None#torch.jit.script(sequential)

        data_dir = '../dataset/p1/labeled/'

        self.threads = 10

        frames_per_clip = 4
        frames_between_clips = 1
        frame_rate = 4

        metadata_location = '../dataset/p1/metadata_labeled_{}{}{}.pt'.format(frames_per_clip, frames_between_clips, frame_rate)
        _precomputed_metadata = None
        ignore_last_metadata = False
        if(isfile(metadata_location) and not ignore_last_metadata):
            _precomputed_metadata = torch.load(metadata_location)

        self.input_toms = sorted(glob.glob(data_dir + '*.mp4'))
        self.video_clips = VideoClips(
            self.input_toms,
            frames_per_clip, #frames_per_clip/clip_length_in_frames
            frames_between_clips, #step_between_clips/frames_between_clips
            frame_rate, #frame_rate
            _precomputed_metadata, #_precomputed_metadata
            num_workers = 0,
            _video_width = 0,
            _video_height = 0,
            _video_min_dimension = 0,
            _audio_samples = 0,
        )
        if(_precomputed_metadata is None):
            torch.save(self.video_clips.metadata, metadata_location)

        folder = 'patched_pr_edge_{}{}{}'.format(frames_per_clip, frames_between_clips, frame_rate)
        os.makedirs(os.path.join('preprocessed', folder), exist_ok=True)
        self.l_mapping = 'preprocessed/{}/l_test_video_{}.pt'.format(folder, '{}')
        self.index_mapping = 'preprocessed/{}/index_test_video.pt'.format(folder)
        self.clips_index = 'preprocessed/{}/clips_test_index.pt'.format(folder)
        self.generate_clips_index()
        #if(not isfile(self.index_mapping)):
        if(len(glob.glob(self.l_mapping.format('*'))) < self.video_clips.num_clips()):
            self.process_threads_videos()
        #else:
        #    self.mapping = torch.load(self.index_mapping)#[:self.num_images*50*50*50]
        #if(not isfile(self.clip_index)):

    def generate_clips_index(self):
        videos = {}
        #Define structeres with videos index and their clips
        for v in range(self.video_clips.num_videos()):
            videos[v] = []
        for clip in range(self.video_clips.num_clips()):
            v, c = self.video_clips.get_clip_location(clip)
            videos[v].append(clip)
        videos_out = {}
        videos_out['videos'] = videos
        videos_out['dim'] = [self.vx, self.vy, self.vz, self.shapes]
        self.index_videos = videos_out['videos']
        torch.save(videos_out, self.clips_index)

    def preprocess_thread_video(self, v):

        for l in tqdm.tqdm(self.index_videos[v]):
            if(isfile(self.l_mapping.format(l))):
                continue

            video, _, _, _ = self.video_clips.get_clip(l)
            #device = torch.device("cuda" if args.cuda else "cpu")
            #device = torch.device("cuda" if True else "cpu")
            #video = video.to(device)
            toy = utils.read_video_shape_gradient(video, self.vx, self.vy, self.vz)

            torch.save(toy, self.l_mapping.format(l))


    def preprocess_thread_clip(self, l):
        if(isfile(self.l_mapping.format(l))):
            return
        #video_idx, clip_idx = self.video_clips.get_clip_location(l)
        video, _, _, _ = self.video_clips.get_clip(l)
        #device = torch.device("cuda" if args.cuda else "cpu")
        #device = torch.device("cuda" if True else "cpu")
        #video = video.to(device)
        toy = utils.read_video_shape_opencv(video, self.vx, self.vy, self.vz)

        torch.save(toy, self.l_mapping.format(l))

        gc.collect()

    #Deprecated
    def generate_index_map(self, samples):
        size = int(self.vx*self.vy*self.vz*self.shapes)
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

    def process_threads_videos(self):
        self.index = 0
        #self.mapping = {}
        p = mp.Pool(self.threads)
        #lst = list(range(self.video_clips.num_videos()))
        lst = list(range(len(self.index_videos)))
        random.shuffle(lst)
        print("Data testing...")
        for i, _ in enumerate(p.imap_unordered(self.preprocess_thread_video, lst, 1)):
            None#sys.stderr.write('\rdone {0:%}'.format(i/self.video_clips.num_clips()))
        print("")
        p.close()
        p.join()

    def process_threads_clips(self):
        self.index = 0
        #self.mapping = {}
        p = mp.Pool(self.threads)
        lst = list(range(self.video_clips.num_clips()))
        random.shuffle(lst)
        print("Data testing...")
        for i, _ in enumerate(p.imap_unordered(self.preprocess_thread_clip, lst, 1)):
            sys.stderr.write('\rdone {0:%}'.format(i/self.video_clips.num_clips()))
        print("")
        p.close()
        p.join()

    def __len__(self):
        #return len(self.mapping)
        return self.video_clips.num_clips()

    def __getitem__(self, index):
        patch_size = None
        clip_out = None

        patch_size = 50
        #Clips shape -> [N x C x W x H] -> [C, W, H, N
        clip = torch.load(self.l_mapping.format(index), map_location='cpu').permute(1, 2, 3, 0).squeeze().float()
        clip_out = rescale(clip, 0.5, anti_aliasing=False, multichannel=True)
        padding = torch.zeros(clip_out.shape)
        clip_out = np.concatenate((padding, clip_out, padding), 2)
        clip_out = np.expand_dims(clip_out, 0)

        return clip_out, patch_size
