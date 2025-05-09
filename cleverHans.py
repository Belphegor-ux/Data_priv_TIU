pip install numpy matplotlib pillow tensorflow cleverhans

import numpy as np
import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

def create_simple_model(input_shape):
   model = tf.keras.Sequential([
       tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
       tf.keras.layers.MaxPooling2D((2, 2)),
       tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
       tf.keras.layers.MaxPooling2D((2, 2)),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(64, activation='relu'),
       tf.keras.layers.Dense(10, activation='softmax')
   ])
   model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
   return model

x = np.load('test_images.npy')
y = np.load('test_labels.npy')

input_shape = x[0].shape
model = create_simple_model(input_shape)

eps = 0.1
adv_x = fast_gradient_method(model, x, eps, np.inf)

logits = model(adv_x)
predictions = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, tf.argmax(y, axis=1)), tf.float32))
print(f"Adversarial accuracy: {accuracy:.4f}")