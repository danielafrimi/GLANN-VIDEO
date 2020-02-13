import numpy as np
import sys
import glo_video_trainer
import utils
import os
import pickle
import data_loader





rn = "golf"
decay = 20 
total_epoch = 200
lr = float(0.01)
factor = float(0.1) 


size = data_loader.DataLoader().get_size()

glo_params = utils.GLOParams(nz=100, force_l2=False)
glo_opt_params = utils.OptParams(lr=lr, factor=factor, batch_size=7, epochs=total_epoch,decay_epochs=decay, decay_rate=0.5)



nt = glo_video_trainer.NAGTrainer(size, glo_params, rn)
nt.train_nag(glo_opt_params)
