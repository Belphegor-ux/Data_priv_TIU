# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:03:12 2025

@author: USER
"""

import numpy as np
import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

# 1. Load MNIST 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, -1)  # shape (num_samples, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

num_classes = 10
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# 2. Build model ---
def create_simple_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_simple_model(x_train.shape[1:])

#  3. Train model ---
model.fit(x_train, y_train_cat, epochs=5, batch_size=64, validation_split=0.1)

#  4. Evaluate on clean test set ---
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Clean test accuracy: {test_acc:.4f}")

#  5. Generate adversarial examples ---
eps = 0.1  # perturbation size
adv_x = fast_gradient_method(model, x_test, eps, np.inf)

# 6. Evaluate on adversarial examples ---
logits = model(adv_x)
predictions = tf.argmax(logits, axis=1)
true_labels = tf.argmax(y_test_cat, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, true_labels), tf.float32))
print(f"Adversarial test accuracy: {accuracy:.4f}")
