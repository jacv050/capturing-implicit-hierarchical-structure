import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from pvae.utils import Constants
from pvae.ops.manifold_layers import GeodesicLayer, MobiusLayer, LogZero, ExpZero, GyroplaneConvLayer

def extra_hidden_layer(hidden_dim, non_lin):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), non_lin)


class EncLinear(nn.Module):
    """ Usual encoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncLinear, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)          # flatten data
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class DecLinear(nn.Module):
    """ Usual decoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecLinear, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)


class EncWrapped(nn.Module):
    """ Usual encoder followed by an exponential map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncWrapped, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)          # flatten data
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class DecWrapped(nn.Module):
    """ Usual encoder preceded by a logarithm map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecWrapped, self).__init__()
        self.data_size = data_size
        self.manifold = manifold
        modules = []
        modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        z = self.manifold.logmap0(z)
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)


class DecGeo(nn.Module):
    """ First layer is a Hypergyroplane followed by usual decoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecGeo, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(GeodesicLayer(manifold.coord_dim, hidden_dim, manifold), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)


class EncMob(nn.Module):
    """ Last layer is a Mobius layers """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncMob, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = MobiusLayer(hidden_dim, manifold.coord_dim, manifold)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))            # flatten data
        mu = self.fc21(e)          # flatten data
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class DecMob(nn.Module):
    """ First layer is a Mobius Matrix multiplication """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecMob, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(MobiusLayer(manifold.coord_dim, hidden_dim, manifold), LogZero(manifold), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)


class DecBernouilliWrapper(nn.Module):
    """ Wrapper for Bernoulli likelihood """
    def __init__(self, dec):
        super(DecBernouilliWrapper, self).__init__()
        self.dec = dec

    def forward(self, z):
        mu, _ = self.dec.forward(z)
        return torch.tensor(1.0).to(z.device), mu

    
################################################################################
#
# Hyperbolic VAE encoders/decoders
#
################################################################################

class EncLinearConv(nn.Module):
    """ 3d convolutional encoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso, posterior=None, num_mixtures=None):
        super(EncLinearConv, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        self.posterior = posterior
        self.num_mixtures = num_mixtures
        modules = []
        modules.append(nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(5, 5, 5), padding=(2, 2, 2)))
        modules.append(nn.ReLU())
        modules.append(nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        modules.append(nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        modules.append(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(5, 5, 5)))
        modules.append(nn.ReLU())
        modules.append(nn.Flatten())
        modules.append(nn.Sequential(nn.Linear(628864, hidden_dim), non_lin))#1024
        self.enc = nn.Sequential(*modules)
        
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x)
        mu = self.fc21(e)          # flatten data
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold #, self.posterior, self.num_mixtures

class DecLinearConv(nn.Module):
    """ 3d convolutional decoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecLinearConv, self).__init__()
        self.data_size = data_size
        self.lin = nn.Sequential(nn.Linear(manifold.coord_dim, 8), non_lin)  # hidden_dim
        
        modules = []
        modules.append(nn.ConvTranspose3d(in_channels=1, out_channels=64, kernel_size=(5, 5, 5), padding=(2, 2, 2)))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        self.dec = nn.Sequential(*modules)
        
        dim_modules = []
        dim_modules.append(nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1), padding=(0, 0, 0)))
        dim_modules.append(nn.ReLU())
        self.dim_reduction = nn.Sequential(*dim_modules)
        
        fc_modules = []
        fc_modules.append(nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=(5, 5, 5)))
        fc_modules.append(nn.ReLU())
        self.fc31 = nn.Sequential(*fc_modules)

    def forward(self, z):
        l = self.lin(z)
        v = l.view(-1, 1, 2, 2, 2)
        r = self.dim_reduction(v)
        d = self.dec(r)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)

class DecLinearVideoConv(nn.Module):
    """ 3d convolutional decoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecLinearVideoConv, self).__init__()
        self.data_size = data_size
        self.lin = nn.Sequential(nn.Linear(manifold.coord_dim, 8), non_lin)  # hidden_dim
        
        modules = []
        modules.append(nn.ConvTranspose3d(in_channels=1, out_channels=64, kernel_size=(5, 5, 5), padding=(2, 2, 2)))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        self.dec = nn.Sequential(*modules)
        
        dim_modules = []
        dim_modules.append(nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1), padding=(0, 0, 0)))
        dim_modules.append(nn.ReLU())
        self.dim_reduction = nn.Sequential(*dim_modules)
        
        fc_modules = []
        fc_modules.append(nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=(20, 20, 70)))
        fc_modules.append(nn.ReLU())
        self.fc31 = nn.Sequential(*fc_modules)

    def forward(self, z):
        l = self.lin(z)
        v = l.view(-1, 1, 2, 2, 2)
        r = self.dim_reduction(v)
        d = self.dec(r)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)

class VideoConv3D(nn.Module):
    def __init__ (self):
        super(VideoConv3D, self).__init__()
        modules = []
        modules.append(nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 3), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        self.vconv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.vconv(x)
        
        return x

class VideoDeConv3D(nn.Module):
    def __init__(self):
        super(VideoDeConv3D, self).__init__()
        modules = []
        modules.append(nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)))
        modules.append(nn.ReLU())
        self.vdeconv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.vdeconv(x)
        return x
    
class EncWrappedConv(nn.Module):
    """ 3D convolutional encoder followed by an exponential map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso, posterior=None, num_mixtures=None):
        super(EncWrappedConv, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        self.posterior = posterior
        self.num_mixtures = num_mixtures
        modules = []
        modules.append(nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(5, 5, 5), padding=(2, 2, 2)))
        modules.append(nn.ReLU())
        modules.append(nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        modules.append(nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        modules.append(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(5, 5, 5)))
        modules.append(nn.ReLU())
        modules.append(nn.Flatten())
        #modules.append(nn.Linear(628864, hidden_dim))
        #modules.append(non_lin)
        modules.append(nn.Sequential(nn.Linear(628864, hidden_dim), non_lin))#1024 11264 628864 9483264
        self.enc = nn.Sequential(*modules)
        
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x)
        """ I
        for layer in self.enc:
            x = layer(x)
            print(x.shape)
        e = x
        """
        mu = self.fc21(e)          # flatten data
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold #, self.posterior, self.num_mixtures

class EncWrappedVideoConv(nn.Module):
    """ 3D convolutional encoder followed by an exponential map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso, posterior=None, num_mixtures=None):
        super(EncWrappedVideoConv, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        self.posterior = posterior
        self.num_mixtures = num_mixtures
        modules = []
        #(10, 10, 9) (9, 9, 9) (8, 8, 8) (8, 8, 8) depth 8
        #(10, 10, 6) (9, 9, 6) (8, 8, 5) (8, 8, 5) depth 4
        modules.append(nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(10, 10, 6), padding=(2, 2, 2)))#30 
        modules.append(nn.ReLU())
        modules.append(nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(9, 9, 6), padding=(1, 1, 1)))#23
        modules.append(nn.ReLU())
        modules.append(nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(8, 8, 5), padding=(1, 1, 1)))#18
        modules.append(nn.ReLU())
        modules.append(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(8, 8, 5)))#14
        modules.append(nn.ReLU())
        modules.append(nn.Flatten())
        modules.append(nn.Sequential(nn.Linear(1024, hidden_dim), non_lin))#1024 11264 628864
        self.enc = nn.Sequential(*modules)
        
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        #print(x.shape)
        e = self.enc(x)
        """ I
        for layer in self.enc:
            x = layer(x)
            print(x.shape)
        e = x
        """
        mu = self.fc21(e)          # flatten data
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold #, self.posterior, self.num_mixtures
    
    
class DecWrappedConv(nn.Module):
    """ 3d convolutional decoder preceded by a logarithm map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecWrappedConv, self).__init__()
        self.data_size = data_size
        self.manifold = manifold
        self.lin = nn.Sequential(nn.Linear(manifold.coord_dim, 8), non_lin)  # hidden_dim
        
        modules = []
        modules.append(nn.ConvTranspose3d(in_channels=1, out_channels=64, kernel_size=(5, 5, 5), padding=(2, 2, 2)))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        self.dec = nn.Sequential(*modules)
        
        dim_modules = []
        dim_modules.append(nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1), padding=(0, 0, 0)))
        dim_modules.append(nn.ReLU())
        self.dim_reduction = nn.Sequential(*dim_modules)
        
        fc_modules = []
        fc_modules.append(nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=(5, 5, 5)))
        fc_modules.append(nn.ReLU())
        self.fc31 = nn.Sequential(*fc_modules)

    def forward(self, z):
        z = self.manifold.logmap0(z)
        l = self.lin(z)
        v = l.view(-1, 1, 2, 2, 2)
        r = self.dim_reduction(v)
        d = self.dec(r)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)
    
class DecGyroConv(nn.Module):
    """ 3d convolutional decoder preceded by a logarithm map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecGyroConv, self).__init__()
        self.data_size = data_size
        self.manifold = manifold
        self.hidden_dim = hidden_dim
        
        gyro_modules = []
        gyro_modules.append(GyroplaneConvLayer(in_features=manifold.coord_dim, out_channels=hidden_dim, kernel_size=1, manifold=manifold))
        gyro_modules.append(nn.ReLU())
        self.gyro_conv = nn.Sequential(*gyro_modules)
        
        dim_modules = []
        dim_modules.append(nn.ConvTranspose3d(in_channels=hidden_dim, out_channels=1, kernel_size=(1, 1, 1), padding=(0, 0, 0))) #300
        dim_modules.append(nn.ReLU())
        self.dim_reduction = nn.Sequential(*dim_modules)
 
        
        modules = []
        modules.append(nn.ConvTranspose3d(in_channels=1, out_channels=64, kernel_size=(5, 5, 5), padding=(2, 2, 2)))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(5, 5, 5), padding=(1, 1, 1)))
        modules.append(nn.ReLU())
        self.dec = nn.Sequential(*modules)
        
        fc_modules = []
        fc_modules.append(nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=(5, 5, 5)))
        fc_modules.append(nn.ReLU())
        self.fc31 = nn.Sequential(*fc_modules)

    def forward(self, z):
        batch = z.shape[1]
        g = self.gyro_conv(z)

        v = self.dim_reduction(g)
        d = self.dec(v)
        mu = self.fc31(d).view(-1, batch, *self.data_size)  # reshape data        
        return mu, torch.ones_like(mu)
    
class DecGyroVideoConv(nn.Module):
    """ 3d convolutional decoder preceded by a logarithm map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecGyroVideoConv, self).__init__()
        self.data_size = data_size
        self.manifold = manifold
        self.hidden_dim = hidden_dim
        
        gyro_modules = []
        gyro_modules.append(GyroplaneConvLayer(in_features=manifold.coord_dim, out_channels=hidden_dim, kernel_size=1, manifold=manifold))
        gyro_modules.append(nn.ReLU())
        self.gyro_conv = nn.Sequential(*gyro_modules)
        
        dim_modules = []
        dim_modules.append(nn.ConvTranspose3d(in_channels=hidden_dim, out_channels=1, kernel_size=(1, 1, 1), padding=(0, 0, 0))) #300
        dim_modules.append(nn.ReLU())
        self.dim_reduction = nn.Sequential(*dim_modules)
 
        
        modules = []
        modules.append(nn.ConvTranspose3d(in_channels=1, out_channels=64, kernel_size=(8, 8, 5), padding=(2, 2, 2))) #(5,5,5) (8,8,8)
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(8, 8, 5), padding=(1, 1, 1))) #(5,5,5) (8,8,8)
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(9, 9, 6), padding=(1, 1, 1))) #(5,5,5) (9,9,9)
        modules.append(nn.ReLU())
        self.dec = nn.Sequential(*modules)
        
        fc_modules = []
        fc_modules.append(nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=(10, 10, 6)))#1024 -> (5, 5, 25) (10,10,10)
        fc_modules.append(nn.ReLU())
        self.fc31 = nn.Sequential(*fc_modules)

    def forward(self, z):
        batch = z.shape[1]
        g = self.gyro_conv(z)
        """ I
        for layer in self.gyro_conv:
            z = layer(z)
            print(z.shape)
        g = z
        """
        v = self.dim_reduction(g)
        d = self.dec(v)
        #*self.data_size
        mu = self.fc31(d).view(-1, batch, *self.data_size)  # reshape data        
        return mu, torch.ones_like(mu)
    
