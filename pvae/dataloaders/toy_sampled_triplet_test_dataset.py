import torch
from torch.utils import data
import pickle
import numpy as np
from skimage.transform import rescale
from os import listdir
from os.path import join

class ToySampledTripletTestDataset(data.Dataset):
    def __init__(self, width, height, depth):
                
        self.width = width
        self.height = height
        self.depth = depth
        
        all_original_toms = []
        toy_data_dir = '../dataset/bio_synthetic/toy_final/test/im_{}.p'
        for i in range(20):
            curr_path = toy_data_dir.format(str(i))
            toy = pickle.load(open(curr_path, 'rb'))
            padded_toy = np.zeros((64, 64, 64))
            padded_toy[8:58, 8:58, 8:58] = toy
            all_original_toms.append(padded_toy.astype(np.float32))
                    
        self.input_toms = all_original_toms
        
        mapping = {}
        index = 0
        
        # 5 x 5
        for l in range(len(all_original_toms)):
            for i in range(6, 56, 1): 
                for j in range(6, 56, 1):
                    for k in range(6, 56, 1): 
                        mapping[index] = (l, i, j, k, 5)
                        index += 1
        
        # 10 x 10
        for l in range(len(all_original_toms)):
            for i in range(4, 54, 1): 
                for j in range(4, 54, 1):
                    for k in range(4, 54, 1): 
                        mapping[index] = (l, i, j, k, 10)
                        index += 1
        
        # 15 x 15
        for l in range(len(all_original_toms)):
            for i in range(0, 50, 1): 
                for j in range(0, 50, 1):
                    for k in range(0, 50, 1): 
                        mapping[index] = (l, i, j, k, 15)
                        index += 1
        
        self.index_mapping = mapping

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, index):    
        
        l, i, j, k, patch_size = self.index_mapping[index]
        
        original_tom = self.input_toms[l]
        
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
