# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:07:03 2025

@author: USER
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from PIL import Image
import os

# --- 1. Load CIFAR-10 ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype(np.float32) / 255.0
y_test = y_test.flatten()
num_classes = 10
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# --- 2. Load or train model ---
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

model = create_simple_model(x_test.shape[1:])
model.fit(x_train, tf.keras.utils.to_categorical(y_train.flatten(), num_classes),
          epochs=10, batch_size=64, validation_split=0.1)

# --- 3. Evaluate on clean test set ---
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Clean test accuracy: {test_acc:.4f}")

# --- 4. Generate adversarial examples ---
eps = 0.05
adv_x = fast_gradient_method(model, x_test, eps, np.inf)

# --- 5. Plot and compare ---
n_samples = 5
os.makedirs('adv_examples', exist_ok=True)

for i in range(n_samples):
    orig = x_test[i]
    adv = adv_x[i].numpy()
    
    # Clip values to [0,1] after attack
    adv = np.clip(adv, 0, 1)
    
    # Plot side by side
    fig, ax = plt.subplots(1, 2, figsize=(4,2))
    ax[0].imshow(orig)
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(adv)
    ax[1].set_title('Adversarial')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()
    
    # Save adversarial image to disk
    adv_img_uint8 = (adv * 255).astype(np.uint8)
    adv_img_pil = Image.fromarray(adv_img_uint8)
    adv_img_pil.save(f'adv_examples/adv_{i}.png')

print(f"✅ Saved {n_samples} adversarial examples to 'adv_examples/' folder")
