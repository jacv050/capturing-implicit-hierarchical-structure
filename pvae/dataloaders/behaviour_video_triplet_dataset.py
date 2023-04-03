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
import time
import os
import tqdm
#import open3d as o3d

mutex = threading.Lock()
#mutexv = threading.Lock()

class BehaviourVideoTripletDataset(data.Dataset):
    def __init__(self, width, height, depth, no_background, np_zeropadding = False):

        self.width = width
        self.height = height
        self.depth = depth

        self.threads = 10
        self.np_zeropadding = np_zeropadding

        #gray scale tensor and resize
        #sequential = torch.nn.Sequential(transforms.Resize((10, 10, 30))) #, ]) #TODO parameterize
        #self.transforms_training = torch.jit.script(sequential)

        #all_inputs = []
        #List of videos
        data_dir = '../dataset/p1/test/'

        frames_per_clip = 4
        frames_between_clips = 4
        frame_rate = 4

        metadata_location = "../dataset/p1/metadata_{}{}{}.pt".format(frames_per_clip, frames_between_clips, frame_rate)
        #metadata_location = "../dataset/p1/metadata.pt"
        _precomputed_metadata = None
        ignore_last_metadata = False
        if(isfile(metadata_location) and not ignore_last_metadata):
            print("Loading metadata...")
            _precomputed_metadata = torch.load(metadata_location)

        self.input_toms = sorted(glob.glob(data_dir + '*.mp4'))
        self.video_clips = VideoClips(
            self.input_toms,
            frames_per_clip, #frames_per_clip/clip_length_in_frames
            frames_between_clips, #step_between_clips/frames_between_clips
            frame_rate, #frame_rate
            _precomputed_metadata, #_precomputed_metadata
            num_workers = 10,
            _video_width = 0,
            _video_height = 0,
            _video_min_dimension = 0,
            _audio_samples=0,
        )
        print(self.input_toms)
        #self.mutex_list = [threading.Lock() for i in range(self.video_clips.num_videos())]
        #self.video_reading = [False for i in range(self.video_clips.num_videos())]

        print(self.video_clips.num_clips())
        if(_precomputed_metadata is None):
            torch.save(self.video_clips.metadata, metadata_location)

        folder = 'patched_pr_edge_{}{}{}'.format(frames_per_clip, frames_between_clips, frame_rate)
        os.makedirs(os.path.join('preprocessed', folder), exist_ok=True)
        #Files after video load and triplet relation
        self.index_mapping = 'preprocessed/{}/index_video_triplets.pt'.format(folder)
        #Files befores triplet after video load
        self.l_mapping = 'preprocessed/{}/l_video_{}.pt'.format(folder, '{}')
        self.no_background = no_background
        
        #if(not isfile(self.index_mapping)):
        self.patch_video_3D()
        #else:
        #    self.mapping = torch.load(self.index_mapping)

        if(len(glob.glob(self.l_mapping.format('*'))) < len(self.mapping)):#num_clips
            self.process_threads_videos(self.input_toms)

    @property
    def metadata(self):
        return self.video_clips.metadata

    # -1 -> Triplet definition require neighbour clips
    def __len__(self):
        return len(self.mapping)

    def patch_video_3D(self):
        videos = {}
        self.mapping = {}
        index = 0

        #Define structeres with videos index and their clips
        for v in range(self.video_clips.num_videos()):
            videos[v] = []
        for clip in range(self.video_clips.num_clips()):
            v, c = self.video_clips.get_clip_location(clip)
            videos[v].append(clip)

        #l_videos = list(range(self.video_clips.num_videos()))
        for v, clips in videos.items():
            size = len(clips)
            if(size < 4):
                continue

            #l_videos.remove(v)
            for i in range(size-2):
                 c1, p, c2 = clips[i], i+1, clips[i+2]
                 #v_aux = random.choice(l_videos)
                 #n = random.randint(0, len(videos[v_aux]))
                 n = clips[size-1] if p <= size/2 else clips[0]
                 self.mapping[index] = (c1, clips[p], c2, n)
                 index += 1
                 
            #l_videos.append(v)
        self.index_videos = videos
        torch.save(self.mapping, self.index_mapping)

    def preprocess_thread_video(self, v):
        
        for l in tqdm.tqdm(self.index_videos[v]):
        
            if(isfile(self.l_mapping.format(l))):
                continue
        
            video_idx, clip_idx = self.video_clips.get_clip_location(l)
            video, _, _ , _ = self.video_clips.get_clip(l)
            #device = torch.device("cuda" if True else "cpu")
            #video = video.to(device)
            
            ind_input = utils.read_video_shape_gradient(video, 50.0, 50.0, 50.0)
            torch.save(ind_input, self.l_mapping.format(l))
            del(video)

            del(ind_input)
            gc.collect()
        
    def preprocess_thread_clip(self, l):
        
        if(isfile(self.l_mapping.format(l))):
            return
        #t1 = time.time()
        #print('Preprocessing: {}'.format(l))
        #video_idx, clip_idx = self.video_clips.get_clip_location(l)
        video, _, _ , _ = self.video_clips.get_clip(l)
        #device = torch.device("cuda" if args.cuda else "cpu")
        #video.to(device)
        ind_input = utils.read_video_shape_opencv(video, 50.0, 50.0, 50.0)

        torch.save(ind_input, self.l_mapping.format(l))
        del(video)

        #TODO Delete this part
        #patch = self.patch_video_3D(l, ind_input)
        del(ind_input)
        gc.collect()
        #t2 = time.time()
        #print(t2-t1)

    def process_threads_clips(self, all_inputs):
        p = mp.Pool(self.threads)
        lst = list(range(self.video_clips.num_clips()))
        random.shuffle(lst)
        print("Data training...")
        for i, _ in enumerate(p.imap_unordered(self.preprocess_thread_clip, lst, 1)):
            sys.stderr.write('\rdone {0:%}'.format(i/self.video_clips.num_clips()))
        sys.stderr.write('\rdone {0:%}'.format(100.0))
        print("")
        p.close()
        p.join()
        gc.collect()

    def process_threads_videos(self, all_inputs):
        p = mp.Pool(self.threads)
        #lst = list(range(self.video_clips.num_videos()))
        lst = list(range(len(self.index_videos)))
        random.shuffle(lst)
        print("Data training...")
        for i, _ in enumerate(p.imap_unordered(self.preprocess_thread_video, lst, 1)):
            None #sys.stderr.write('\rdone {0:%}'.format(i/self.video_clips.num_clips()))
        #sys.stderr.write('\rdone {0:%}'.format(1))
        print("")
        p.close()
        p.join()
        gc.collect()

    def __getitem__(self, index):
        #t1 = time.time()
        cl, p, cr, n = self.mapping[index]
        r = random.randint(0, 5)
        n = n + r if n == 0 else n-r

        #Clips shape -> [x,y,z] -> [W, H, N]
        #Clips shape -> [N x C x W x H] -> [C, W, H, N]
        clip_l = torch.load(self.l_mapping.format(cl)).permute(1, 2, 3, 0).squeeze().float() #clip left
        clip_p = torch.load(self.l_mapping.format(p)).permute(1, 2, 3, 0).squeeze().float() #clip positive
        clip_r = torch.load(self.l_mapping.format(cr)).permute(1, 2, 3, 0).squeeze().float() #clip right
        clip_n = torch.load(self.l_mapping.format(n)).permute(1, 2, 3, 0).squeeze().float() #clip negative

        clip_l = rescale(clip_l, 0.5, anti_aliasing=False, multichannel=True)#0.2
        clip_p = rescale(clip_p, 0.5, anti_aliasing=False, multichannel=True)#0.2
        clip_r = rescale(clip_r, 0.5, anti_aliasing=False, multichannel=True)#0.2
        clip_n = rescale(clip_n, 0.5, anti_aliasing=False, multichannel=True)#0.2
        
        #Or. Version
        x = np.concatenate((clip_l, clip_p, clip_r), 2) #[W, H, N] [10, 10, 30] [25,25,75]
        x = np.expand_dims(x, 0)

        if(self.np_zeropadding):
            #print("ZEROPADDED")
            padding = torch.zeros(clip_p.shape)
            clip_p = np.concatenate((padding, clip_p, padding), 2)
            clip_n = np.concatenate((padding, clip_n, padding), 2)
        
        clip_p = np.expand_dims(clip_p, 0)
        clip_n = np.expand_dims(clip_n, 0)

        #t2 = time.time()
        #print("{}".format(t2-t1))
        return x, clip_p, clip_n, -1

