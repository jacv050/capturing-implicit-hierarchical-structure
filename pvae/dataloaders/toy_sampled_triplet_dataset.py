import torch
from torch.utils import data
import pickle
import numpy as np
from skimage.transform import rescale
from os import listdir
from os.path import join

class ToySampledTripletDataset(data.Dataset):
    def __init__(self, width, height, depth, no_background):
                
        self.width = width
        self.height = height
        self.depth = depth
        
        
        all_inputs = []
        toy_data_dir = '/pasteur/data/hierarchical-toy-dataset/toy_irregular4/test/im_{}.p'
        for i in range(20):
            curr_path = toy_data_dir.format(str(i))
            toy = pickle.load(open(curr_path, 'rb'))
            all_inputs.append(toy.astype(np.float32))
                    
        self.input_toms = all_inputs
        
        mapping = {}
        index = 0
        
        for l in range(len(all_inputs)):
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
                                                  
                        if no_background:
                            ind_input = all_inputs[l]
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
                                
                                mapping[index] = ((l, i, j, k, sampled_patch_size),
                                                  (l, i, j, k, child_patch_size),
                                                  (l, negative_i, negative_j, negative_k, child_patch_size))
                                index += 1
                                
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
                                
                            mapping[index] = ((l, i, j, k, sampled_patch_size),
                                              (l, i, j, k, child_patch_size),
                                              (l, negative_i, negative_j, negative_k, child_patch_size))
                            index += 1
        
        self.index_mapping = mapping

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, index):     
        
        ((l, i, j, k, sampled_patch_size),
         (l, i, j, k, child_patch_size),
         (l, negative_i, negative_j, negative_k, child_patch_size)) = self.index_mapping[index]
        
        sampled_patch_margin = int(sampled_patch_size / 2)
        child_patch_margin = int(child_patch_size / 2) 
        
        ind_input = self.input_toms[l]
        
        parent_patch = ind_input[i-sampled_patch_margin:i+sampled_patch_margin+1,
                                 j-sampled_patch_margin:j+sampled_patch_margin+1,
                                 k-sampled_patch_margin:k+sampled_patch_margin+1]
        
        positive_child = ind_input[i-child_patch_margin:i+child_patch_margin+1,
                                   j-child_patch_margin:j+child_patch_margin+1,
                                   k-child_patch_margin:k+child_patch_margin+1]
        
        
        negative_child = ind_input[negative_i-child_patch_margin:negative_i+child_patch_margin+1,
                                   negative_j-child_patch_margin:negative_j+child_patch_margin+1,
                                   negative_k-child_patch_margin:negative_k+child_patch_margin+1]
        
        parent_patch = rescale(parent_patch, 10.0/(sampled_patch_margin * 2 + 1), anti_aliasing=False)
        parent_patch = np.expand_dims(parent_patch, 0)
        
        positive_child = rescale(positive_child, 10.0/(child_patch_margin * 2 + 1), anti_aliasing=False)
        positive_child = np.expand_dims(positive_child, 0)
        
        negative_child = rescale(negative_child, 10.0/(child_patch_margin * 2 + 1), anti_aliasing=False)
        negative_child = np.expand_dims(negative_child, 0)
        
        return parent_patch, positive_child, negative_child, (1.0 - ((sampled_patch_margin * 2 + 1)/35.0))
    