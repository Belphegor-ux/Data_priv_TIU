# -*- coding: utf-8 -*-
"""
Created on Fri May  9 21:58:28 2025

@author: USER
"""

import numpy as np
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#scale 0,1
x_test = x_test.astype(np.float32) / 255.0
x_test = np.expand_dims(x_test, axis=-1)  #something sample(num_samples, 28, 28, 1)


num_classes = 10
y_test_one_hot = np.zeros((y_test.shape[0], num_classes))
y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1


np.save('test_images.npy', x_test)
np.save('test_labels.npy', y_test_one_hot)

print("Saved MNIST test_images.npy and test_labels.npy")
