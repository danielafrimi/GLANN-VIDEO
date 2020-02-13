import imageio
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image
import utils



GOLF_DATA_LISTING = '/cs/labs/yedid/daniel023/cifar_glo_clean/golf_small.txt'
DATA_ROOT = '/cs/labs/yedid/daniel023/cifar_glo_clean'
#path_write = "/cs/labs/yedid/daniel023/cifar_glo_clean/golf_real.txt" 

class DataLoader(object):
    def __init__(self, batch_size=7):
        # reading data list
        self.batch_size = batch_size
        self.crop_size = 64
        self.frame_size = 32
        self.image_size = 128
        #self.file = open(path_write, "w")

        # Shuffle video index.
        data_list_path = os.path.join(GOLF_DATA_LISTING)  # 603776 video path
        with open(data_list_path, 'r') as f:
            self.video_index = [x.strip() for x in f.readlines()]
            np.random.shuffle(self.video_index)

        self.size = len(self.video_index) # number of videos

        # A pointer in the dataset
        self.cursor = 0

    def get_batch(self, i, rp):
        t_out = torch.zeros((self.batch_size, self.frame_size, 3, self.crop_size, self.crop_size))
        to_tensor = transforms.ToTensor()  # Transforms 0-255 numbers to 0 - 1.0.
        index = rp[i * self.batch_size : (i + 1) * self.batch_size]

        for i in range(len(index)):
            video_path = os.path.join(DATA_ROOT, self.video_index[index[i]])
            inputimage = cv2.imread(video_path)
            count = inputimage.shape[0] / self.image_size
            for j in range(self.frame_size): # until 32
                if j < count:
                    cut = int(j * self.image_size)
                else:
                    cut = int(count - 1) * self.image_size
                crop = inputimage[cut: cut + self.image_size, :]
                temp_out = to_tensor(cv2.resize(crop, (self.crop_size, self.crop_size)))
                # temp_out = temp_out * 2 - 1 # for normalize to [-1,1]
                t_out[i,j,:,:] = temp_out
            # utils.make_gif(t_out.permute(0,2,1,3,4), "/cs/labs/yedid/daniel023/cifar_glo_clean/movie",self.batch_size)
        return t_out, index



    def get_size(self):
        return self.size


    def shuffle_data(self):
        return np.random.permutation(self.size)

    def shuffle_(self):
        return np.arange(self.size)




   # def copy_relavant(self):
   #     counter = 0
   #     for i in range(self.size):
   #         video_path = os.path.join(DATA_ROOT, self.video_index[self.cursor])
   #         inputimage = cv2.imread(video_path)
   #         if(inputimage is None):
   #             self.cursor += 1
   #             continue
   #         counter  = counter +1
   #         if counter == 50000:
   #             break
   #         print(self.video_index[self.cursor] + "\n" + "counter is", counter)
   #         self.file.write(self.video_index[self.cursor] + "\n")
   #         self.cursor += 1
   #     print("num of videos is:", counter)
   #     self.file.close()

