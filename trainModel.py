# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:54:00 2020

@author: Hai Li
"""

"""
#To load data files onto colab 
from google.colab import files
uploaded = files.upload()

import zipfile
import io
data = zipfile.ZipFile(io.BytesIO(uploaded['Data.zip']), 'r')
data.extractall()

data.printdir()
"""

import os
#import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = 'Data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'train')

# Directory of training images of NOK
train_0_dir = os.path.join(train_dir, '0')

# Directory of traning images of OK
train_1_dir = os.path.join(train_dir, '1')

# Use the same training data for validation
# Set shuffle = True to make training and validation data different
validation_0_dir = os.path.join(validation_dir, '0')
validation_1_dir = os.path.join(validation_dir, '1')

import cv2

im = cv2.imread('Data/train/0/17.jpg')
#im = cv2.resize(im,(128,72),interpolation = cv2.INTER_AREA)

print("type of image:", type(im))
# <class 'numpy.ndarray'>

print(f"image shape = {im.shape}, image size = {im.size}")
print(f"the color of the first pixel of the image is {im[1,1]}")


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(108, 108, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-3),
              metrics=['acc'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(108, 108),  # All images will be resized to 150x150
        batch_size=16,
        shuffle = True,
        class_mode='binary')

# Flow validation images in batches using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(108, 108),
        batch_size=16,
        shuffle = True,
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=1,  # images = batch_size * steps
      epochs=15,
      validation_data=validation_generator,
      validation_steps=1,  # images = batch_size * steps
      verbose=2)

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
    

#save the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


#predict
import os
import numpy as np
from keras.preprocessing import image
i=1
while i < 100:
    img_name = "Data/train/1/"+str(i)+".jpg"
    if os.path.exists(img_name):
        img = image.load_img(img_name, target_size=(108, 108))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        images = images/255
        classes = model.predict(images, batch_size=1)
        print(classes)
        if classes[0,0] < 0.5:
            print(f"{i} is NOK")
        else:
            print(f"{i} is OK")
        i+=1
    else:
        i+=1
        continue