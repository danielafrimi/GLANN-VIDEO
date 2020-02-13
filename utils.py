from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import vgg_metric
import model
import resnet
import imageio


GLOParams = collections.namedtuple('NAGParams', 'nz ngf mu sd force_l2')
GLOParams.__new__.__defaults__ = (None, None, None, None, None)
GANParams = collections.namedtuple('GANParams', 'ndf weight_d')
GANParams.__new__.__defaults__ = (None, None, None)
OptParams = collections.namedtuple('OptParams', 'lr factor ' +'batch_size epochs ' + 'decay_epochs decay_rate lr_ratio')
OptParams.__new__.__defaults__ = (None, None, None, None, None, None)
ImageParams = collections.namedtuple('ImageParams', 'sz nc n mu sd')
ImageParams.__new__.__defaults__ = (None, None, None)




# Receives [3,32,64,64] tensor, and creates a gif
def make_gif(images, filename, batch_size):
    img = []
    frames = []
    for i in range(batch_size):
        x = images[i].permute(1, 2, 3, 0)
        x = x.cpu().detach().numpy()
        img.append(x)

    for i in range(batch_size):
        for j in range(32):
            frames += [img[i][j]]
        imageio.mimwrite(filename + str(i) + ".gif", frames, format='GIF-FI')
        frames.clear()

def logger(file, directory, batch_size, lr, factor,  epochs, VGG, numBlocks):
    print("Start Training GLO")
    file.write("Directory in: " +  str(directory) + '\n')
    file.write("Batch Size is: " + str(batch_size) + '\n')
    file.write("Learning rate is: " + str(lr) + '\n')
    file.write("factor is: " + str(factor) + '\n')
    file.write("Num epochs: " + str(epochs) + '\n')
    if VGG:
        file.write("Peceptual loss : VGG"  + '\n')
    else:
        file.write("Peceptual loss : resneXt-" + str(numBlocks)  + '\n')
    file.close()



#maybe to do permute to get (32, batch_size, 3, 64,64) and than pass the frames send it to loss function???????????????????????????
def get_frames(vid):
    img = []
    frames = []
    frames_batch = []
    for i in range(10):
        x = vid[i].permute(1, 2, 3, 0)
        x = x.cpu().detach().numpy()
        img.append(x)

    for i in range(10):
        for j in range(32):
            frames += [img[i][j]] #TODO check visulize (vgg)
        frames_batch.append(frames)
    return frames_batch




def get_frames_new(vid):
    frames = []
    vid_new = vid.permute(2,0,1,3,4)
    for i in range(32):
        frames.append(vid_new[i])
    return frames


def distance_metric(sz, nc, force_l2=False):
    # return vgg_metric._VGGFixedDistance()
    if force_l2:
        return nn.L1Loss().cuda()
    if sz == 16:
        return vgg_metric._VGGDistance(2)
    elif sz == 32:
        return vgg_metric._VGGDistance(3)
    elif sz == 64:
        return vgg_metric._VGGDistance(4)
    elif sz > 64:
        return vgg_metric._VGGMSDistance()


def generator(sz, nc, nz, ngf):
    return resnet.Generator(nz, sz, nfilter=ngf, nfilter_max=512)


def discriminator(sz, nc, ndf):
    if sz == 16:
        return model._netD_16(ndf, nc)
    elif sz == 32:
        return model._netD_32(ndf, nc)
    elif sz == 64:
        return model._netD_64(ndf, nc)
    elif sz == 256:
        return model._netD_256(ndf, nc)
    else:
        print("Invalid image size")
        sys.exit()


def sample_gaussian(x, m):
    x = x.data.numpy()
    mu = x.mean(0).squeeze()
    cov2 = np.cov(x, rowvar=0)
    z = np.random.multivariate_normal(mu, cov2, size=m)
    z_t = torch.from_numpy(z).float()
    radius = z_t.norm(2, 1).unsqueeze(1).expand_as(z_t)
    z_t = z_t / radius
    return Variable(z_t.cuda())


def unnorm(ims, mu, sd):
    for i in range(len(mu)):
        ims[:, i] = ims[:, i] * sd[i]
        ims[:, i] = ims[:, i] + mu[i]
    return ims


def format_im(ims_gen, mu, sd):
    if ims_gen.size(1) == 3:
        rev_idx = torch.LongTensor([2, 1, 0]).cuda()
    elif ims_gen.size(1) == 1:
        rev_idx = torch.LongTensor([0]).cuda()
    else:
        arr = [i for i in range(ims_gen.size(1))]
        rev_idx = torch.LongTensor(arr).cuda()
    # Generated images
    ims_gen = unnorm(ims_gen, mu, sd)
    ims_gen = ims_gen.data.index_select(1, rev_idx)
    ims_gen = torch.clamp(ims_gen, 0, 1)
    return ims_gen


def denorm(x):
    out = (x + 1.0) / 2.0
    return nn.Tanh(out)