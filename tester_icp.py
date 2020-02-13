import numpy as np
import sys
import torch
import torch.nn as nn
import icp
import torchvision.utils as vutils
import modelvideo
import utils


counter = 1
rn = "golf"
nz = 100
parallel = True
W = torch.load('runs%d/nets_%s/netZ_glo.pth' % (counter, rn))
W = W['emb.weight'].data.cpu().numpy()

netG = modelvideo.netG_new(nz).cuda()

if torch.cuda.device_count() > 1:
    parallel = True
    print("Using", torch.cuda.device_count(), "GPUs!")
    netG = nn.DataParallel(netG)

state_dict = torch.load('runs%d/nets_%s/netG_glo.pth' % (counter, rn))
netG.load_state_dict(state_dict) # load the weights for generator (GLO)
if parallel:
    netG = netG.module

# d is the dimension of noise vector (e)
d = 16
nepoch = 50
icpt = icp.ICPTrainer(W, d)
icpt.train_icp(nepoch)
torch.save(icpt.icp.netT.state_dict(), 'runs%d/nets_%s/netT_nag.pth' % (counter, rn)) #saves the param of the netT

#Prediction
z = icpt.icp.netT(torch.randn(100, d).cuda())
print("z shape", z.shape)
video = netG(z)
print("video shape is", video.shape)


# utils.make_gif(utils.denorm(video), 'runs3/ims_%s/sample' % (rn), 1)
utils.make_gif(video, 'runs%d/ims_%s/sample' % (counter, rn), 5)