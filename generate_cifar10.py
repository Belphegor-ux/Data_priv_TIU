# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:00:00 2025

@author: USER
"""

import numpy as np
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# make images
x_test = x_test.astype(np.float32) / 255.0  # shape (num_samples, 32, 32, 3)

y_test = y_test.flatten()

num_classes = 10
y_test_one_hot = np.zeros((y_test.shape[0], num_classes))
y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1


np.save('test_images.npy', x_test)
np.save('test_labels.npy', y_test_one_hot)

print("Saved CIFAR-10 test_images.npy and test_labels.npy")
