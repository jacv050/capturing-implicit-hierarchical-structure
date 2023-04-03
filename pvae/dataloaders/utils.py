import torch
from torchvision import transforms
from torchvision.datasets.video_utils import VideoClips
from math import ceil
import gc
from scipy import ndimage, misc

import skimage
import numpy as np
import cv2 as cv2

sequential = torch.nn.Sequential(transforms.Grayscale())
transform = torch.jit.script(sequential)

def conv_3dmean(jx, jy, jz, input3d, size_x, size_y, size_z):
    output = torch.zeros(size_x, size_y, size_z)
    for xi in range(size_x):
        for yi in range(size_y):
            for zi in range(size_z):
                #mean of subsquare
                #cell = input3d[xi*jx:jx,yi*jy:jy,zi*jz:jz].mean()
                z = zi*jz
                x = xi*jx
                y = yi*jy
                m = input3d[z:z+jz, :, x:x+jx, y:y+jy]
                if(m.shape[0] != 0 and m.shape[2] != 0 and m.shape[3] != 0):
                    cell = m.mean()
                    #print(cell)
                    output[xi,yi,zi] = cell
    return output

#def read_video_shape(l, videoclips, size_x, size_y, size_z, transf=None):
def read_video_shape(video, size_x, size_y, size_z, transf=None, conv_mean=False):
    #video, audio, info = torchvision.io.read_video(path, start_pts=0, end_pts=None, pts_unit='sec')
    #video, audio, info, video_idx = videoclips.get_clip(l)
    #resize to box
    #video_squared = self.transforms(video)
    #[N x W x H x C] -> [N x C x W x H]
    video = video.permute((0, 3, 1, 2))
    #jump_x = ceil(video.shape[2]/size_x)
    #jump_y = ceil(video.shape[3]/size_y)
    #jump_z = ceil(video.shape[0]/size_z)
    #convolved_video = []
    #print(v.shape)
    if(transf is None):
        video_t = transform(video)
        video_t = transforms.Resize((int(size_x), int(size_y)))(video_t)
    else:
        video_t = transf(video)

    #convolved_video = conv_3dmean(jump_x, jump_y, jump_z, video_t.float(), int(size_x), int(size_y), int(size_z))
    #convolved_video = video_t if not conv_mean else conv_3dmean(jump_x, jump_y, jump_z, video_t.float(), int(size_x), int(size_y), int(size_z))
    return video_t

def read_video_shape_opencv(video, size_x, size_y, size_z, transf=None, conv_mean=False):
    #N C W H
    video_t = torch.zeros((video.shape[0], 1, int(size_y), int(size_x)))
    #[N x W x H x C] -> [N x H x W x C]
    video = video.permute((0, 2, 1, 3))
    for frame in range(video.shape[0]):
        #fr = cv.CreateMat(frame.shape[1], frame.shape[2], cv.CV_32FC3)
        #fr = cv.fromarray()
        #gray = cv2.cvtColor(frame, cv2.RGB2GRAY)
        canny = cv2.Canny(video[frame].numpy(), 100, 200)
        video_t[frame] = torch.from_numpy(cv2.resize(canny, (int(size_y), int(size_x))))
        
    return video_t

def read_video_shape_gradient(video, size_x, size_y, size_z, transf=None, conv_mean=False):
    #N C W H
    video_t = torch.zeros((video.shape[0], 1, int(size_y), int(size_x)))
    #[N x W x H x C] -> [N x H x W x C]
    video = video.permute((0, 2, 1, 3))
    for frame in range(video.shape[0]):
        grayscale = skimage.color.rgb2gray(video[frame].numpy())
        resized = skimage.transform.resize(grayscale, (int(size_y), int(size_x)))        
        sx = ndimage.sobel(resized, axis=0, mode='constant')
        sy = ndimage.sobel(resized, axis=1, mode='constant')

        gradient = np.hypot(sx, sy)
        
        video_t[frame, 0] = torch.from_numpy(gradient)
        
    return video_t