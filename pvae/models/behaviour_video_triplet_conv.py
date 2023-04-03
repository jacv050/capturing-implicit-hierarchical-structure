import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms
from skimage.transform import rescale
import sys

import math
from numpy import prod
from .vae import VAE
from pvae.utils import Constants

from pvae.distributions import RiemannianNormal, WrappedNormal
from torch.distributions import Normal

import time
from pvae import manifolds
from .architectures import EncLinear, DecLinear, EncWrapped, DecWrapped, EncMob, DecMob, DecGeo, DecBernouilliWrapper, EncWrappedConv, DecWrappedConv, EncLinearConv, DecLinearConv, DecGyroConv, VideoConv3D, VideoDeConv3D, DecGyroVideoConv, EncWrappedVideoConv, DecLinearVideoConv

#from pvae.dataloaders.toy_sampled_triplet_dataset import ToySampledTripletDataset
#from pvae.dataloaders.toy_sampled_triplet_test_dataset import ToySampledTripletTestDataset
from pvae.dataloaders.behaviour_video_triplet_dataset import BehaviourVideoTripletDataset
from pvae.dataloaders.behaviour_video_triplet_test_dataset import BehaviourVideoTripletTestDataset

width = 25#10 #25
height = 25#10 #25
depth = 12#30 #75
data_size = torch.Size([1, width, height, depth])

class BehaviourVideoTripletConv(VAE):
    def __init__(self, params):
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        manifold = getattr(manifolds, params.manifold)(params.latent_dim, c)
        super(BehaviourVideoTripletConv, self).__init__(
            eval(params.prior),   # prior distribution
            eval(params.posterior),   # posterior distribution
            dist.Normal,        # likelihood distribution
            eval('Enc' + params.enc)(manifold, data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim, params.prior_iso),
            eval('Dec' + params.dec)(manifold, data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim),
            params
        )
        BehaviourVideoTripletConv.np_zeropadding = params.np_zeropadding
        BehaviourVideoTripletConv.no_padding = params.no_padding
        
        self.vconv = VideoConv3D()
        self.vdeconv = VideoDeConv3D()
        self.manifold = manifold
        self.c = c
        self._pz_mu = nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)
        self.modelName = 'Toy'
            
        '''for child in self.dec.children():
            for param in child.parameters():
                param.requires_grad = False'''
    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std), self.manifold

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 6, 'pin_memory': True} if device == "cuda" else {}

        #train_loader = DataLoader(    
        #    ToySampledTripletDataset(width, height, depth, no_background=True),
        #    batch_size=batch_size, shuffle=True, **kwargs)
        train_loader = DataLoader(
            BehaviourVideoTripletDataset(width, height, depth, no_background=False, 
                np_zeropadding=BehaviourVideoTripletConv.np_zeropadding),
                batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(
            BehaviourVideoTripletTestDataset(width, height, depth, 
                no_padding=BehaviourVideoTripletConv.no_padding),
                batch_size=batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N, K = 64, 9
        folder = 'patched_pr_{}{}{}'.format(8, 1, 8)
        self.l_mapping = 'preprocessed/{}/l_test_video_{}.pt'.format(folder, '1')
        
        clip = torch.load(self.l_mapping, map_location='cpu').permute(1, 2, 3, 0).squeeze().float()
        clip_out = rescale(clip, 0.5, anti_aliasing=False, multichannel=True)
        mean, means, samples = super(BehaviourVideoTripletConv, self).generate(N, K)
        save_image(mean.data.cpu(), '{}/gen_mean_{:03d}.png'.format(runPath, epoch))
        save_image(means.data.cpu(), '{}/gen_means_{:03d}.png'.format(runPath, epoch))

    def reconstruct(self, data, runPath, epoch):
        print(data[:8].shape)
        recon = super(BehaviourVideoTripletConv, self).reconstruct(data[:2])
        comp = torch.cat([data[:8], recon])
        save_image(comp.data.cpu(), '{}/recon_{:03d}.png'.format(runPath, epoch))
        
    def getMus(self, data, runPath):
        d = data
        if(not self.np_zeropadding):
            d = self.vconv(d)
        mus = self.enc(d)
        return mus[0]
        
    def forward(self, x, positive_child, negative_child, K=1):
        #t1 = time.time()
        if(not self.np_zeropadding):
            x=self.vconv(x)
        #x=self.vconv(x)
        qz_x = self.qz_x(*self.enc(x))#posterior
       
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        #px_z = self.vdeconv(px_z)
        
        if positive_child == None:
            return qz_x, px_z, zs

        parent_mu = self.enc(x)[0]
        positive_child_mu = self.enc(positive_child)[0]
        negative_child_mu = self.enc(negative_child)[0]
        
        #t2 = time.time()
        #print("{}".format(t2-t1))
        return qz_x, px_z, zs, parent_mu, positive_child_mu, negative_child_mu
