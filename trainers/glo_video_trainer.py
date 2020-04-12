from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import shutil
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import utils
import modelvideo
from torchvision.transforms import ToPILImage
import imageio
import perceptual_loss_video
import data_loader
import torchvision.utils as vutils
import model_video_orig
import glo as lap


counter = 25
load = False
VGG = True
LAP = False

# GLO
class NAG():
    # NAGParams(nz=dim of z, ngf=None, mu=None, sd=None, force_l2=False)
    def __init__(self, glo_params, vid_params, rn):
        self.netZ = model_video_orig._netZ(glo_params.nz, vid_params.n)
        self.netZ.apply(model_video_orig.weights_init) # init the weights of the model
        self.netZ.cuda() # on GPU
        self.rn = rn
        self.lr = 0.01
        self.data_loader = data_loader.DataLoader()
        self.netG = model_video_orig.netG_new(glo_params.nz)
        self.netG.apply(model_video_orig.weights_init)
        self.netG.cuda()

        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            print("Using " + str(num_devices) + " GPU's")
            for i in range(num_devices):
                print(torch.cuda.get_device_name(i))
            self.netG = nn.DataParallel(self.netG)

        if load: #Load point
            self.load_weights(counter, self.rn)

        self.vis_n = 100
        fixed_noise = torch.FloatTensor(self.vis_n,glo_params.nz).normal_(0, 1) # for visualize func - Igen
        self.fixed_noise = fixed_noise.cuda()
        self.nag_params = glo_params
        self.vid_params = vid_params
        self.blockResnext = 101


        if VGG:
            self.dist_frame = utils.distance_metric(64, 3, glo_params.force_l2)
        elif LAP:
            self.lap_loss = lap.LapLoss(max_levels=3)
        else:
            self.dist = perceptual_loss_video._resnext_videoDistance(self.blockResnext)


    def train(self,  opt_params, train_details, vis_epochs=10):
        arr_layers = [5, 8, 12, 15, 18, 21, 24, 26, 27, 31]
        print("Directory is:", counter)
        utils.logger(train_details, counter, opt_params.batch_size, self.lr, opt_params.factor, opt_params.epochs, VGG, self.blockResnext)
        for epoch in range(opt_params.epochs):
            er = self.train_epoch(epoch, opt_params, vis_epochs, arr_layers)
            print("GLO Epoch: %d Error: %f" % (epoch, er))
            torch.save(self.netZ.state_dict(), 'runs%d/nets_%s/netZ_glo.pth' % (counter, self.rn)) #saves the params of the network Z
            torch.save(self.netG.state_dict(), 'runs%d/nets_%s/netG_glo.pth' % (counter, self.rn)) #saves the params of the network G
            if epoch % vis_epochs == 0:
                self.visualize(epoch, opt_params)

    def train_epoch(self,  epoch, opt_params, vis_epochs, arr_layers):

        rp = self.data_loader.shuffle_data() #TODO  - nonDeteminstic

        batch_size = opt_params.batch_size
        batch_n = self.vid_params.n // batch_size
        visualize_flag = True

        # Compute learning rate
        decay_steps = epoch // opt_params.decay_epochs
        lr = opt_params.lr * opt_params.decay_rate ** decay_steps

        # Initialize optimizers
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr * opt_params.factor,betas=(0.5, 0.999))
        optimizerZ = optim.Adam(self.netZ.parameters(), lr=lr, betas=(0.5, 0.999))


        # Start optimizing
        er = 0
        for i in range(batch_n):
            vid, np_idx_new = self.data_loader.get_batch(i, rp)
            videos = vid.permute(0,2,1,3,4).float().cuda()
            idx = torch.from_numpy(np_idx_new).long().cuda() # create tensor out of np_idx, and then pass to gpu since it is an input to the net

            # Forward pass
            self.netZ.zero_grad() # zeros the grad, since we want to calculate the gradients only according to the current batch.
            self.netG.zero_grad()
            zi = self.netZ(idx) # Output of the net.
            Vi = self.netG(zi)
            # normal_video = (2 * Vi) - 1
            frames_real = utils.get_frames_new(videos)
            frames_fake = utils.get_frames_new(Vi) #TODO check this

            if VGG:
                rec_loss = self.dist_per_frame(frames_real, frames_fake) #TODO train with VGG frames

            elif LAP:
                rec_loss = self.lap_per_frame(frames_real, frames_fake) #TODO train with LAPLACIAN frames
            else:
                # Reconstruction loss - resnext-101 perptcual
                rec_loss = self.dist(Vi, videos) #TODO train with resnext

            # rec_loss = torch.abs(Vi - videos) #L1 loss

            rec_loss = rec_loss.mean()
            # Backward pass - compute gradient of the loss with respect to model parameters
            rec_loss.backward()

            # optimization step - update G and Z (perform a single optimization step (parameter update))
            optimizerG.step()
            optimizerZ.step()

            # update training loss
            er += rec_loss.item()

            if (epoch % vis_epochs == 0) and visualize_flag and (epoch is not 0): #TODO if i dont do suffle what to see?
                # Irec = self.netG(self.netZ(idx)) # reconstructed video
                # utils.make_gif(Vi, 'runs%d/ims_%s/reconstructions_epoch_%03d' % (counter, self.rn, epoch), opt_params.batch_size)
                # utils.make_gif(videos, 'runs%d/ims_%s/actual_epoch_%03d' % (counter, self.rn, epoch), opt_params.batch_size) # actual videos
                self.save_frames_check(frames_fake, arr_layers, epoch, 'runs%d/frames_vis_%s/reconstructions_epoch_%03d')
                self.save_frames_check(frames_real, arr_layers, epoch, 'runs%d/frames_vis_%s/actual_epoch_%03d')
                visualize_flag = False

        self.netZ.get_norm()
        # calculate average losses
        er = er / batch_n
        return er

    def save_frames_check(self, frames, arr_layers, epoch, path):
        for i in range(len(arr_layers)):
            vutils.save_image(frames[arr_layers[i]], (path + str(i)  + ".png") % (counter, self.rn, epoch), normalize=False)


    def dist_per_frame(self, frames_real, frames_fake):
        rec_loss = 0
        for i in range(31):
            frame_loss = self.dist_frame(frames_fake[i], frames_real[i]).mean() #TODO check this one, it was without the mean(1)
            rec_loss = rec_loss + frame_loss
        return rec_loss

    def lap_per_frame(self, frames_real, frames_fake):
        rec_loss = 0
        for i in range(31):
            frame_loss = self.lap_loss(frames_fake[i], frames_real[i]).mean() #TODO check this one, it was without the mean(1)
            rec_loss = rec_loss + frame_loss
        return rec_loss



    def visualize(self, epoch, opt_params):
        Igen = self.netG(self.fixed_noise) # GLO on a noise
        utils.make_gif(Igen, 'runs%d/ims_%s/generations_epoch_%03d' % (counter, self.rn, epoch), opt_params.batch_size)

        z = utils.sample_gaussian(self.netZ.emb.weight.clone().cpu(),self.vis_n)
        Igauss = self.netG(z) # GLO on gaussian
        utils.make_gif(Igauss, 'runs%d/ims_%s/gaussian_epoch_%03d' % (counter, self.rn, epoch), opt_params.batch_size)

        # idx = torch.from_numpy(np.arange(opt_params.batch_size)).long().cuda()
        # Irec = self.netG(self.netZ(idx)) # reconstructed video
        # utils.make_gif(Irec, 'runs%d/ims_%s/reconstructions_epoch_%03d' % (counter, self.rn, epoch), opt_params.batch_size)

        # Iact = vid_np[:self.vis_n].permute(0,2, 1, 3, 4).cuda() # actual images
        # self.make_gif(Iact, 'runs/ims_%s/act%03d' % (self.rn, epoch), opt_params.batch_size)

    def load_weights(self,counter, rn):
        state_dict = torch.load('runs%d/nets_%s/netG_glo.pth' % (counter, rn))
        self.netG.load_state_dict(state_dict) # load the weights for generator (GLO)
        state_dict = torch.load('runs%d/nets_%s/netZ_glo.pth' % (counter, rn))
        self.netZ.load_state_dict(state_dict) # load the weights for generator (GLO)

class NAGTrainer():

    def __init__(self, n, glo_params, rn):
        self.sz = 64 #resolution
        self.rn = rn
        self.nc = 3
        self.n = n #size of the train set
        self.image_params = utils.ImageParams(sz=self.sz, nc=self.nc, n=self.n)
        self.nag = NAG(glo_params, self.image_params, rn)


        if not os.path.isdir("runs%d" % counter):
            os.mkdir("runs%d" % counter)
        shutil.rmtree("runs%d/ims_%s" % (counter, self.rn), ignore_errors=True)
        # shutil.rmtree("nets", ignore_errors=True)
        self.file = open("runs%d/train_details.txt" % counter, 'w')

        os.mkdir("runs%d/ims_%s" % (counter,self.rn))

        if not os.path.isdir("runs%d/frames_vis_%s" % (counter,self.rn)):
            os.mkdir("runs%d/frames_vis_%s" % (counter,self.rn))

        if not os.path.isdir("runs%d/nets_%s" % (counter, self.rn)):
            os.mkdir("runs%d/nets_%s" % (counter, self.rn))

    def train_nag(self, opt_params): # opt_paran is all the params for learning (lr,epochs, decay and more)
        self.nag.train( opt_params, self.file) # pass the trainer the train images and the param for learning
