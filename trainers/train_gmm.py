import numpy as np
import sys
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import utils
import torchvision.utils as vutils
import model
import pickle
from torch.utils.data import TensorDataset
import os
import model_video_orig

def main():
    counter = 20
    rn = "golf"
    nz = 100
    parallel = True

    W = torch.load('runs%d/nets_%s/netZ_glo.pth' % (counter, rn))
    W = W['emb.weight'].data.cpu().numpy()

    netG = model_video_orig.netG_new(nz).cuda()
    if torch.cuda.device_count() > 1:
        parallel = True
        print("Using", torch.cuda.device_count(), "GPUs!")
        netG = nn.DataParallel(netG)


    Zs = utils.sample_gaussian(torch.from_numpy(W), 10000)
    Zs = Zs.data.cpu().numpy()


    state_dict = torch.load('runs%d/nets_%s/netG_glo.pth' % (counter, rn))
    netG.load_state_dict(state_dict) # load the weights for generator (GLO)
    if parallel:
        netG = netG.module


    gmm = GaussianMixture(n_components=100, covariance_type='full', max_iter=100, n_init=10)
    gmm.fit(W)

    z = torch.from_numpy(gmm.sample(100)[0]).float().cuda()
    video = netG(z)
    utils.make_gif(video, 'runs%d/ims_%s/sample' % (counter, rn), 16)
    return video

if __name__ == "__main__":
    video = main()
