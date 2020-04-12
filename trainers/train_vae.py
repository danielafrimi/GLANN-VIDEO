import numpy as np
import sys
import torch
import torch.nn as nn
import vae
import utils
import torchvision.utils as vutils
import model
import model_video_orig


def main():
    counter = 21
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

    state_dict = torch.load('runs%d/nets_%s/netG_glo.pth' % (counter, rn))
    netG.load_state_dict(state_dict) # load the weights for generator (GLO)

    if parallel:
        netG = netG.module


    d = 16
    nepoch = 200
    vaet = vae.VAETrainer(W, d)
    vaet.train_vae(nepoch)
    torch.save(vaet.vae.netVAE.state_dict(), 'runs%d/nets_%s/netVAE.pth' % (counter, rn))

    z = vaet.vae.netVAE.decode(torch.randn(100, d).cuda())
    video = netG(z)
    print("video shape is", video.shape)

    utils.make_gif(video, 'runs%d/ims_%s/sample' % (counter, rn), 16)
    return video

if __name__ == "__main__":
    video = main()
