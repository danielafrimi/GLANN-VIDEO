from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch


# The weights_init function takes an initialized model as input and reinitializes all convolutional,
# convolutional-transpose, and batch normalization layers to meet this criteria. This function is
# applied to the models immediately after initialization
def weights_init(m): # m is a model
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Emb') != -1:
        init.normal_(m.weight, mean=0, std=0.01)


class _netZ(nn.Module):
    # nz - length of latent vector (the size of each embedding vector - dim)
    # n - size of z_i (num_embeddings)
    def __init__(self, nz, n):
        super(_netZ, self).__init__()
        self.n = n
        self.emb = nn.Embedding(self.n, nz)
        self.nz = nz

    def get_norm(self):
        wn = self.emb.weight.norm(2, 1).data.unsqueeze(1)
        self.emb.weight.data = \
            self.emb.weight.data.div(wn.expand_as(self.emb.weight.data))

    def forward(self, idx): #idx is in shape of batch size
        z = self.emb(idx).squeeze()
        return z


class _background(nn.Module):
    def __init__(self, nz):
        super(_background, self).__init__()
        self.conv1_background = nn.ConvTranspose2d(nz,512,4,bias=True)
        self.bn_in_background = nn.BatchNorm2d(512)


        self.conv2_background = nn.ConvTranspose2d(512, 512, 4, 2,1,  bias=True)
        self.bn_conv3_background = nn.BatchNorm2d(512)

        self.bn_conv4_background = nn.BatchNorm2d(256)
        self.bn_conv5_background = nn.BatchNorm2d(128)
        self.non_lin_background = nn.ReLU()
        self.tanh_background = nn.Tanh()

        self.expand_conv2 = nn.ConvTranspose2d(512, 512, 1, bias=True) 
        self.conv3_background = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=True)
        self.expand_conv4 = nn.ConvTranspose2d(128, 128, 1, bias=True) 
        self.conv4_background = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=True)
        self.conv5_background = nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=True)

    def forward(self, z):
        zn = z.norm(2, 1).detach().unsqueeze(1).expand_as(z)
        z = z.div(zn)
        z = z.view(-1, 100, 1, 1)
        z = self.conv1_background(z)
        z = self.bn_in_background(z)
        z = self.non_lin_background(z)
        z = self.conv2_background(z)
        z = self.bn_conv3_background(z)
        z = self.non_lin_background(z)

        z= self.expand_conv2(z) 
        z = self.non_lin_background(z)
        z = self.conv3_background(z)
        z = self.bn_conv4_background(z)
        z = self.non_lin_background(z)
        z = self.conv4_background(z)
        z= self.bn_conv5_background(z)
        z = self.non_lin_background(z)

        z = self.expand_conv4(z) 
        z = self.non_lin_background(z) 
        z= self.bn_conv5_background(z)

        z = self.conv5_background(z)
        z = self.tanh_background(z)
        return z

class G_video(nn.Module):
    def __init__(self, nz):
        super(G_video, self).__init__()
        self.conv1 = nn.ConvTranspose3d(nz, 512, (2,4,4) ,bias=True)
        self.conv2 = nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=True)
        self.conv3 = nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=True) #changed here to 256
        self.conv4 = nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=True)
        self.expand_conv2 = nn.ConvTranspose3d(256, 256, 1 , bias=True) #TODO check expand
        self.batch_norm1 = nn.BatchNorm3d(512)
        self.batch_norm2 = nn.BatchNorm3d(256)
        self.batch_norm3 = nn.BatchNorm3d(128)
        self.batch_norm4 = nn.BatchNorm3d(64)
        self.nonlin = nn.ReLU()

    def forward(self, z):
        zn = z.norm(2, 1).detach().unsqueeze(1).expand_as(z)
        z = z.div(zn)
        z = z.view(-1, 100, 1, 1, 1)
        z = self.conv1(z)
        z = self.batch_norm1(z)
        z = self.nonlin(z)
        z = self.conv2(z)
        z = self.batch_norm2(z)
        z = self.nonlin(z)
        z = self.expand_conv2(z)
        z = self.nonlin(z)
        z = self.batch_norm2(z)
        z = self.conv3(z)
        z = self.batch_norm3(z)
        z = self.nonlin(z)
        z = self.conv4(z)
        z = self.batch_norm4(z)
        z = self.nonlin(z)
        return z


class netG_new(nn.Module):
    def __init__(self, nz):
        super(netG_new, self).__init__()
        self.background = _background(nz)
        self.video = G_video(nz)
        self.foreground = nn.Sequential(nn.ConvTranspose3d(64, 3, 4, 2, 1, bias=True), nn.Tanh())
        self.mask = nn.Sequential(nn.ConvTranspose3d(64, 1, 4, 2, 1, bias=True), nn.Sigmoid())

    def forward(self, z):
        background = self.background(z)
        video = self.video(z)
        foreground = self.foreground(video)
        mask = self.mask(video)
        mask_repeated = mask.repeat(1, 3, 1, 1, 1)  # repeat for each color channel. [-1, 3, 32, 64, 64]
        background_frames = background.unsqueeze(2).repeat(1, 1, 32, 1, 1)  # [-1,3,32,64,64]
        out = torch.mul(mask, foreground) + torch.mul(1 - mask, background_frames)
        return out



