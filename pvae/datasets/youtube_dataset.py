import json
from collections import OrderedDict
from os.path import isfile
import os
import torch
import glob
import ntpath
from torch.utils import data
from torchvision.datasets.video_utils import VideoClips

#"dataset_info/finegym/finegym_annotation_info_v1.1.json"
#"dataset_info/finegym/categories/set_categories.txt"
class YoutubeDataset(data.Dataset):
    def __init__(self, annotation_file):
        data_dir = '../dataset/p1/labeled/'
        
        #448
        self.frames_per_clip = 4 #4#20
        self.frames_between_clips = 1#20 8
        self.frame_rate = 4#4

        metadata_location = '../dataset/p1/metadata_labeled_{}{}{}.pt'.format(self.frames_per_clip, self.frames_between_clips, self.frame_rate)
        _precomputed_metadata = None
        ignore_last_metadata = False
        if(isfile(metadata_location) and not ignore_last_metadata):
            _precomputed_metadata = torch.load(metadata_location)
        self.input_toms = sorted(glob.glob(data_dir + '*.mp4'))
        self.video_clips = VideoClips(
            self.input_toms,
            self.frames_per_clip, #frames_per_clip/clip_length_in_frames
            self.frames_between_clips, #step_between_clips/frames_between_clips
            self.frame_rate, #frame_rate
            _precomputed_metadata, #_precomputed_metadata
            num_workers = 0,
            _video_width = 0,
            _video_height = 0,
            _video_min_dimension = 0,
            _audio_samples = 0,
        )
        
        if (_precomputed_metadata is None):
            torch.save(self.video_clips.metadata, metadata_location)
        #We index clips in our way
        self.generate_clips_index(self.video_clips.metadata)

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f, object_pairs_hook=OrderedDict)
            
        self.videos_labeled = self.preprocess()
    
    """
    Lee las etiquetas del json y calcula las etiquetas para las secciones 
    """        
    def preprocess(self): 
        clip_time = self.frames_per_clip/float(self.frame_rate)
        step_time = self.frames_between_clips/float(self.frame_rate)
        #We look for the downloaded videos in the annotation file
        videos_labeled = []
        for video_id, list_clips in self.video_index.items():
            labels_clips = []
            for i in range(len(list_clips)):
                clip_id = list_clips[i]
                #Calculate the times clip
                init_time = step_time * i #init clip
                end_time = init_time + clip_time #end clip
                clip_label = -1
                for event_id, dict_event in self.annotations[video_id].items():
                    event = event_id.split('_')
                    init_event = int(event[1])
                    end_event = int(event[2])
                    #Que empiece después del inicio
                    if(init_time < init_event or init_time > end_event):
                        continue
                    for segment_id, dict_segment in dict_event["segments"].items():
                        segment = segment_id.split('_')
                        init_action = int(segment[1]) + init_event #Action time is inside event_time
                        end_action = int(segment[2]) + init_event
                        #if(init_time < init_action or init_time > end_action):
                        #The action should happen after the beggining of event. This is checked before.
                        #if(init_time > init_action and init_time < end_action):
                        if(init_time >= init_action and end_time <= end_action):
                            clip_label = dict_segment["label"]
                            break
                labels_clips.append(clip_label)
            videos_labeled.append(labels_clips)   

        return videos_labeled
    
    """
    Cambia la estructura interna de los metadatos de VideClips para tener
    una relación directa entre video_id y los clips que lo componen
    output = OrderedDict()
    output[video_id]/output[url_youtube]
    output[video_id] = list_of_clips_video_id
    """                    
    def generate_clips_index(self, metadata):
        videos = OrderedDict()
        #Define structers with videos index and their clips
        list_video_id = [] #list with youtube ids
        for v in range(self.video_clips.num_videos()):
            video_path = metadata["video_paths"][v]
            video_id = ntpath.basename(video_path)[0:-8]
            list_video_id.append(video_id)
            videos[video_id] = []
        for clip in range(self.video_clips.num_clips()):
            v, c = self.video_clips.get_clip_location(clip)
            videos[list_video_id[v]].append(c)

        self.video_index = videos

    def save(self, output_file):
        torch.save(self.videos_labeled, output_file)
        
    def save_by_video(self, output_dir):
        output_file = "truth_all_{}.pt"
        for i in range(len(self.videos_labeled)):
            v = self.videos_labeled[i]
            torch.save(v, os.path.join(output_dir, output_file.format(i)))
        with open(os.path.join(output_dir, "video_index.json"), 'w') as f:
            json.dump([k for k, v in self.video_index.items()], f)
            
