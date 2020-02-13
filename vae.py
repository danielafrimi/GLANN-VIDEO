from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import faiss
# from fbpca import pca
import collections
from torch.nn import functional as F


OptParams = collections.namedtuple('OptParams', 'lr batch_size epochs ' +
                                                'decay_epochs decay_rate ')
OptParams.__new__.__defaults__ = (None, None, None, None, None)


class _netVAE(nn.Module):
    def __init__(self, xn, yn):
        super(_netVAE, self).__init__()

        self.e_1 = nn.Linear(xn, 512)
        self.e_1_bn = nn.BatchNorm1d(512)
        self.e_2 = nn.Linear(512, 512)
        self.e_2_bn = nn.BatchNorm1d(512)
        self.e_mu = nn.Linear(512, yn)
        self.e_logvar = nn.Linear(512, yn)

        self.d_1 = nn.Linear(yn, 512)
        self.d_1_bn = nn.BatchNorm1d(512)
        self.d_2 = nn.Linear(512, 512)
        self.d_2_bn = nn.BatchNorm1d(512)
        self.d_out = nn.Linear(512, xn)

    def encode(self, x):
        if False:
            h1 = F.relu(self.e_1_bn(self.e_1(x)))
            h2 = F.relu(self.e_2_bn(self.e_2(h1)))
        else:
            h1 = F.relu(self.e_1(x))
            h2 = F.relu(self.e_2(h1))
        mu = self.e_mu(h2)
        logvar = self.e_logvar(h2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        if False:
            h1 = F.relu(self.d_1_bn(self.d_1(z)))
            h2 = F.relu(self.d_2_bn(self.d_2(h1)))
        else:
            h1 = F.relu(self.d_1(z))
            h2 = F.relu(self.d_2(h1))
        z = self.d_out(h2)
        if False:
            zn = z.norm(2, 1).data.unsqueeze(1)
            z.data = z.data.div(zn.expand_as(z.data))
        else:
            z = z / z.norm(2, 1).unsqueeze(1)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class _VAE():
    def __init__(self, e_dim, z_dim):
        # super(_VAE, self).__init__()
        self.e_dim = e_dim
        self.z_dim = z_dim
        self.netVAE = _netVAE(e_dim, z_dim).cuda()

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + 0.01 * KLD

    def train(self, z_np, opt_params):
        self.opt_params = opt_params
        for epoch in range(opt_params.epochs):
            self.train_epoch(z_np, epoch)

    def train_epoch(self, z_np, epoch):
        # Compute batch size
        batch_size = self.opt_params.batch_size
        n, d = z_np.shape
        batch_n = n // batch_size
        rp = np.random.permutation(n)

        # Compute learning rate
        decay_steps = epoch // self.opt_params.decay_epochs
        lr = self.opt_params.lr * self.opt_params.decay_rate ** decay_steps
        # Initialize optimizers
        optimizerT = optim.Adam(self.netVAE.parameters(), lr=lr,
                                betas=(0.5, 0.999))
        criterion = nn.MSELoss().cuda()
        self.netVAE.train()

        # Start optimizing
        er = 0
        rp = np.random.permutation(len(z_np))

        for i in range(batch_n):
            self.netVAE.zero_grad()
            idx_np = rp[i * batch_size + np.arange(batch_size)]
            z_act = torch.from_numpy(z_np[idx_np]).float().cuda()
            z_est, mu, logvar = self.netVAE(z_act)
            loss = self.loss_function(z_est, z_act, mu, logvar)
            loss.backward()
            er += loss.item()
            optimizerT.step()

        print("Epoch: %d Error: %f" % (epoch, er / batch_n))


class VAETrainer():
    def __init__(self, f_np, d):
        self.f_np = f_np
        self.vae = _VAE(f_np.shape[1], d)

    def train_vae(self, n_epochs):
        uncca_opt_params = OptParams(lr=1e-3, batch_size=128, epochs=n_epochs,
                                     decay_epochs=50, decay_rate=0.5)
        self.vae.train(self.f_np, uncca_opt_params)
