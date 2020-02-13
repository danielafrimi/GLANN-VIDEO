import torch
import torch.nn as nn
from resnext.opts import parse_opts
from resnext import model
import argparse
import easydict


def get_args(depth):
    if depth == 101:
        path = 'resnext-101-kinetics.pth'
        type = 'resnext'
    if depth == 50:
        path = 'resnet-50-kinetics.pth'
        type = 'resnet'

    args = easydict.EasyDict(
        {
            "root_path": '/root/data/ActivityNet',
            "video_path": 'video_kinetics_jpg',
            "annotation_path": 'kinetics.json',
            "result_path": 'results',
            "dataset": 'kinetics',
            "n_classes": 400,
            "n_finetune_classes": 400,
            "sample_size": 64,
            "sample_duration": 32,
            "initial_scale": 1.0,
            "n_scales": 5,
            "scale_step": 0.84089641525,
            "train_crop": 'corner',
            "learning_rate": 0.1,
            "momentum": 0.9,
            "dampening": 0.9,
            "weight_decay": 1e-3,
            "mean_dataset": 'activitynet',
            "no_mean_norm": False,
            "std_norm": False,
            "nesterov": False,
            "optimizer": 'sgd',
            "lr_patience": 10,
            "batch_size": 32,
            "n_epochs": 100,
            "begin_epoch": 1,
            "n_val_samples": 3,
            "resume_path": '',
            "pretrain_path": path,
            "ft_begin_index": 0,
            "no_train": False,
            "no_val": False,
            "test": False,
            "test_subset": 'val',
            "scale_in_test": 1.0,
            "crop_position_in_test": 'c',
            "no_softmax_in_test": False,
            "no_cuda": False,
            "n_threads": 4,
            "checkpoint": 10,
            "no_hflip": False,
            "norm_value": 1,
            "model": type,
            "model_depth": depth,
            "resnet_shortcut": 'B',
            "wide_resnet_k": 2,
            "resnext_cardinality": 32,
            "manual_seed": 1
        }
    )
    return args


class _resnext_videoDistance(nn.Module):
    def __init__(self, depth):
        super(_resnext_videoDistance, self).__init__()
        self.resnext = _resnextFeatures(depth)

    def forward(self, video1, video2):
        batch_size = video1.size(0)
        f1 = self.resnext(video1)
        f2 = self.resnext(video2)
        loss = torch.abs(video1 - video2).view(batch_size, -1).mean(1).cuda()

        for i in range(1, 5):
            layer_loss = torch.abs(f1[i] - f2[i]).view(batch_size, -1).mean(1)
            loss = loss + layer_loss
        return loss


class _resnextFeatures(nn.Module):
    def __init__(self, depth):
        super(_resnextFeatures, self).__init__()
        args = get_args(depth)
        self._resnext = model.generate_model(args)[0]


    def forward(self, video):
        return self._resnext(video) #return an array of outputs of some layers in resnext-101




