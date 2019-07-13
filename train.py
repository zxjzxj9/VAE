#! /usr/bin/env python


import tensorflow as tf

from model import VAEModel
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='VAE parameters')
parser.add_argument('--batch-size', type=int, default=128, help="batch size")
parser.add_argument('--num-latent', type=int, default=128, help="number of latent variables")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--klscale', type=float, default=1e-6, help="balance factor between KL-divergence and MSE loss")
opt = parser.parse_args()

class Trainer(object):

    def __init__(self, lr=1e-4, **kwargs):
        self.lr = lr
        self.vae = VAEModel(**kwargs)
        self.data_path = "/datasets/Img/img_align_celeba/img_align_celeba"


    def train(self):

        step = 0
        with self.vae.g_train.as_default():
            kl, mse = self.vae.train_graph(self.data_path)
            loss = kl + mse

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

            tf.summary.scalar('KLLoss', kl)
            tf.summary.scalar('MSELoss', mse)
            tf.summary.scalar('TotalLoss', loss)

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("/tmp/log", self.vae.g_train)

            saver = tf.train.Saver()
            with tf.Session() as s:
                s.run(tf.global_variables_initializer())

                while True:
                    try:
                        _, kl_d, mse_d, loss_d, mgd = s.run([optim, kl, mse, loss, merged])
                        step += 1
                        writer.add_summary(mgd, step)

                        if step % 1000 == 0:
                            print("Saving model...")
                            saver.save(s, "./saved_models/model.ckpt")
                        print("Iter: {:10d}, KLLoss: {:12.6f}, MSELoss: {:12.6f}, TotalLoss: {:12.6f}".format(step, kl_d, mse_d, loss_d), end="\n")
                    except tf.errors.OutOfRangeError:
                        print("Training finished...")
                        break


if __name__ == "__main__":
    tr = Trainer(lr=opt.lr, bs=opt.batch_size, n_latent=opt.num_latent, klscale=opt.klscale)
    tr.train()
