#! /usr/bin/env python

import tensorflow as tf
import glob
import os
import numpy as np

kinit_func = lambda: tf.initializers.glorot_normal()

class VAEModel(object):
    def __init__(self, bs=128, n_latent=128, klscale=1e-6):
        self._init_graph()

        self.bs = bs
        self.n_latent = n_latent
        self.klscale = klscale

    def _init_graph(self):
        self.g_train_ = tf.Graph()
        self.g_valid_ = tf.Graph()
        self.g_infer_ = tf.Graph()

    @property
    def g_train(self):
        return self.g_train_

    @property
    def g_valid(self):
        return self.g_valid_

    @property
    def g_infer(self):
        return self.g_infer_

    def encode(self, inputs):

        with tf.name_scope("encoder") as scope:
            # 256 -> 32
            layer1 = tf.layers.conv2d(inputs, 32, 3, strides=2, padding="same", kernel_initializer=kinit_func())
            layer1 = tf.layers.batch_normalization(layer1)
            layer1 = tf.nn.relu(layer1)
            layer1 = tf.layers.conv2d(layer1, 32, 4, strides=4, padding="same", kernel_initializer=kinit_func())
            # 32 -> 16
            layer2 = tf.layers.conv2d(layer1, 64, 3, strides=2, padding="same", kernel_initializer=kinit_func())
            layer2 = tf.layers.batch_normalization(layer2)
            layer2 = tf.nn.relu(layer2)
            #layer2 = tf.layers.conv2d(layer2, 128, 4, strides=4, padding="same", kernel_initializer=kinit_func())
            # 16 -> 8
            layer3 = tf.layers.conv2d(layer2, 128, 3, strides=2, padding="same", kernel_initializer=kinit_func())
            layer3 = tf.layers.batch_normalization(layer3)
            layer3 = tf.nn.relu(layer3)
            # 8 -> 4
            layer4 = tf.layers.conv2d(layer3, 256, 3, strides=2, padding="same", kernel_initializer=kinit_func())
            layer4 = tf.layers.batch_normalization(layer4)
            layer4 = tf.nn.relu(layer4)
            # 4 -> 2
            layer5 = tf.layers.conv2d(layer4, 512, 3, strides=2, padding="same", kernel_initializer=kinit_func())
            layer5 = tf.layers.batch_normalization(layer5)
            layer5 = tf.nn.relu(layer5)
            # 2 -> 1
            layer6 = tf.layers.conv2d(layer5, 1024, 3, strides=2, padding="same", kernel_initializer=kinit_func())
            #layer6 = tf.layers.batch_normalization(layer6)
            layer6 = tf.nn.relu(layer6)
            layer6 = tf.reshape(layer6, shape=(-1, 1024))

            mu = tf.layers.dense(layer6, self.n_latent)
            logsigma = tf.layers.dense(layer6, self.n_latent)

            return mu, logsigma

    def decode(self, inputs):

        with tf.name_scope("decoder") as scope:
            # 1 -> 4
            layer1 = tf.layers.conv2d_transpose(inputs, 1024, 4, strides=4, padding="same", kernel_initializer=kinit_func())
            layer1 = tf.layers.batch_normalization(layer1)
            layer1 = tf.nn.relu(layer1)
            # 4 -> 16
            layer2 = tf.layers.conv2d_transpose(layer1, 512, 4, strides=4, padding="same", kernel_initializer=kinit_func())
            layer2 = tf.layers.batch_normalization(layer2)
            layer2 = tf.nn.relu(layer2)
            # 16 -> 32
            layer3 = tf.layers.conv2d_transpose(layer2, 256, 4, strides=2, padding="same", kernel_initializer=kinit_func())
            layer3 = tf.layers.batch_normalization(layer3)
            layer3 = tf.nn.relu(layer3)
            # 32 -> 64
            layer4 = tf.layers.conv2d_transpose(layer3, 128, 4, strides=2, padding="same", kernel_initializer=kinit_func())
            layer4 = tf.layers.batch_normalization(layer4)
            layer4 = tf.nn.relu(layer4)
            # 64 -> 128
            layer5 = tf.layers.conv2d_transpose(layer4, 64, 4, strides=2, padding="same", kernel_initializer=kinit_func())
            layer5 = tf.layers.batch_normalization(layer5)
            layer5 = tf.nn.relu(layer5)
            # 128 -> 256
            layer6 = tf.layers.conv2d_transpose(layer5, 32, 4, strides=2, padding="same", kernel_initializer=kinit_func())
            layer6 = tf.layers.batch_normalization(layer6)
            layer6 = tf.nn.relu(layer6)

            layer7 = tf.layers.conv2d(layer6, 3, 1, strides=1, padding="same", kernel_initializer=kinit_func())
            layer7 = tf.nn.tanh(layer7)
            return layer7

    def train_graph(self, folder):

        def process_fn(f):
            img = tf.read_file(f)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [256, 256])
            img = img*2.0/255.0 - 1.0
            return img

        with self.g_train_.as_default():
            img_paths = glob.glob(os.path.join(folder, "*.jpg"))
            img_paths = [imgf for imgf in img_paths if os.stat(imgf).st_size != 0]
            #print(img_paths[:1000])
            images = tf.data.Dataset.from_tensor_slices(img_paths).map(process_fn)
            images = images.repeat(100000).shuffle(1000).batch(self.bs)

            iterator = images.make_one_shot_iterator()
            images = iterator.get_next()
            #return iterator, images

            mu, logsigma = self.encode(images)
            sigma = tf.exp(logsigma)
            KLLoss = 0.5*tf.reduce_sum(mu**2 + sigma**2 - 2.0*logsigma - 1)
            KLLoss = KLLoss*self.klscale

            ## reparameteration
            #prior = tf.placeholder(shape=(None, self.n_latent), dtype=tf.float32, name="priors")
            prior = tf.random_normal(tf.shape(sigma))
            pinp = prior*sigma + mu
            preshape = tf.reshape(pinp, shape=(-1, 1, 1, self.n_latent))

            dec = self.decode(preshape)

            MSELoss = 0.5*tf.reduce_mean((dec - images)**2)
            tf.summary.image('InpImg', (images + 1)/2.0, max_outputs=4)
            tf.summary.image('GenImg', (dec + 1)/2.0, max_outputs=4)
            return KLLoss, MSELoss


if __name__ == "__main__":
    vae = VAEModel()

    prior, kl, mse = vae.train_graph("/datasets/Img/img_align_celeba/img_align_celeba")
    #_, images = vae.train_graph("/datasets/Img/img_align_celeba/img_align_celeba")
    with vae.g_train.as_default():
        with tf.Session() as s:
            s.run(tf.global_variables_initializer())
            kld, msed = s.run([kl, mse], feed_dict={prior: np.random.randn(128, 128)})
            print(kld, msed)
    #print(kl, mse)
    #inputs = tf.placeholder(shape=(None, 256, 256, 3), dtype=tf.float32, name="images")
    #print(vae.encode(inputs))
    #print(vae.decode(vae.encode(inputs)))
