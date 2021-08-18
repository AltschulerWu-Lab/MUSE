'''
Inference of inception-v3 model with pretrained parameters on ImageNet
'''


import tensorflow.compat.v1 as tf

# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()
import tensorflow_hub as hub
import numpy as np
import cv2
import pandas as pd

# Load saved inception-v3 model
module = hub.Module("./inception_v3")

# images should be resized to 299x299
input_imgs = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
features = module(input_imgs)

# Provide the file indices
# This can be changed to image indices in strings or other formats
spot_info = pd.read_csv('spot_info.csv', header=0, index_col=None)
image_no = spot_info.shape[0]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    img_all = np.zeros([image_no, 299, 299, 3])

    # Load all images and combine them as a single matrix
    # This loop can be changed to match the provided image information
    for i in range(image_no):
        # Here, all images are stored in example_img and in *.npy format
        # if using image format, np.load() can be replaced by cv2.imread()
        file_name = './example_img/' + spot_info.iloc[i, 2] + '.npy'
        temp = np.load(file_name)
        temp2 = temp.astype(np.float32) / 255.0
        img_all[i, :, :, :] = temp2

    # Check if the image are loaded successfully.
    if (i == image_no - 1):
        print('+++Successfully load all images+++')
    else:
        print('+++Image patches missing+++')

    # Input combined image matrix to Inception-v3 and output last layer as deep feature
    fea = sess.run(features, feed_dict={input_imgs: img_all})

    # Save inferred image features
    np.save('Inception_img_feature.npy', fea)

  