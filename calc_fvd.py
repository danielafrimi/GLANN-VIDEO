#@article{unterthiner2018towards,
#title={Towards Accurate Generative Models of Video: A New Metric \& Challenges},
# author={Unterthiner, Thomas and van Steenkiste, Sjoerd and Kurach, Karol and Marinier, Raphael and Michalski, Marcin and Gelly, Sylvain},
# journal={arXiv preprint arXiv:1812.01717},year={2018}}

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataloader
import tensorflow as tf
import frechet_video_distance as fvd
import train_vae
import utils


# Number of videos must be divisible by 16.
NUMBER_OF_VIDEOS = 16
VIDEO_LENGTH = 32
rn = "golf"



data_loader = dataloader.DataLoader()
rp = data_loader.shuffle_data()
vid, np_idx_new = data_loader.get_batch(0, rp)

videos = vid.permute(0,1,3,4,2).float()
videos = vid.permute(0,1,3,4,2).float().cpu().detach().numpy()
videos = tf.convert_to_tensor(videos, dtype=tf.float32)


generated = train_vae.main() #Choose a model
generated_vid = generated.permute(0,1,3,4,2).float().cpu().detach().numpy()
generated_vid = tf.slice(generated_vid,[0,0,0,0,0], [NUMBER_OF_VIDEOS,32,64,64,3])
generated_vid = tf.convert_to_tensor(generated_vid, dtype=tf.float32)

result = fvd.calculate_fvd(
    fvd.create_id3_embedding(fvd.preprocess(videos, (224, 224))),
    fvd.create_id3_embedding(fvd.preprocess(generated_vid, (224, 224))))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print("FVD is: %.2f." % sess.run(result))
