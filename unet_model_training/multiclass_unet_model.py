"""
This multiclass UNet model example can allow us to also process RGB data
from the TetraNet devices as well.

"""



# import os
from PIL import Image, ImageOps
import numpy as np
from numpy import asarray
from matplotlib import pyplot as plt
from keras.utils import normalize
import os

from google.colab import drive
drive.mount('/content/drive') # run if not mounted already

%tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# pet_images & trimaps
drive_dir = '/content/drive/MyDrive/Project TetraNet/files/Oxford-IIT/'
# !ls '/content/drive/MyDrive/Project TetraNet/files/Oxford-IIT/trimaps'

# ONLY RUN THIS ONCE, COMMENT OUT ONCE YOU UNZIP THE FILES
!unzip "/content/drive/MyDrive/Project TetraNet/files/Oxford-IIT/images.zip" -d "/content"
!unzip "/content/drive/MyDrive/Project TetraNet/files/Oxford-IIT/masks.zip" -d "/content"

train_dir = '/content/images/'
mask_images_dir = '/content/trimaps/'


# pair all of the names
training_names = os.listdir(training_datset)
mask_image_names = os.listdir(mask_images_dir)

for i in training_names:
  if (i.split('.')[1] == 'mat'):
    training_names.remove(i)

training_names = sorted(training_names)
mask_image_names = sorted(mask_image_names)

c = 0
for x,y in zip(training_names, mask_image_names):
  if (x.split('.')[0] != y.split('.')[0]):
    c += 1

# 0 means good to go
print(str(c) + ' errors')
print(len(training_names), len(mask_image_names))

image_dataset = []
mask_dataset = []

ai, bm, errors = 0, 0, 0

while (bm < len(mask_image_names)):
  image_name = training_names[ai]
  mask_name = mask_image_names[bm]
  # sanity check to make sure everything works
  image = Image.open(training_datset + image_name).convert('RGB')
  image = image.resize((128, 128))
  image = asarray(image)
  # check to see if correct sizes
  try:
    x,y,z = image.shape
    if (not (x == 128 and y == 128 and z == 3)):
      errors += 1
      ai += 1
      bm += 1
  except Exception as inst:
    print('oog')
    errors += 1
    ai += 1
    bm += 1
  # all images should be correct shape here
  image_dataset.append(image)
  # open and reformat all masks to grayscale
  mask = Image.open(mask_images_dir + mask_name)
  mask = ImageOps.grayscale(mask)
  mask = mask.resize((128,128))
  mask = asarray(mask)
  mask_dataset.append(mask)
  # increment pointers
  ai += 1
  bm += 1

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

import random
# cmap docs: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
num = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[num], cmap='viridis')
plt.axis('off')
plt.subplot(122)
plt.imshow(mask_dataset[num])
plt.axis('off')


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

# converts into already normalized images (can accept float32 or float64) / 255.0 -- normalized range [0,1]
def unet_model(IMGH, IMGW, CHANNELS, CLASSES):

  # begin with 1st layer input = 128x128x3 aka. 128x128 RGB image (3 channels)
  inputs = Input((IMGH, IMGW, CHANNELS))
  s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
  s = inputs

  c1 = Conv2D(filters= 16, kernel_size=3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(s)
  c1 = Dropout(rate= 0.1)(c1)
  c1 = Conv2D(filters= 16, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c1)

  # move into 2nd layer : convolution 1
  m1 = MaxPooling2D(pool_size= 2)(c1)
  # m1 becomes input to c2
  c2 = Conv2D(filters= 32, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(m1)
  c2 = Dropout(rate= 0.1)(c2)
  c2 = Conv2D(filters= 32, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c2)

  # move into 3rd layer : convolution 2
  m2 = MaxPooling2D(pool_size= 2)(c2)
  c3 = Conv2D(filters= 64, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(m2)
  c3 = Dropout(rate= 0.2)(c3)
  c3 = Conv2D(filters= 64, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c3)

  # move into 4th layer : convolution 3
  m3 = MaxPooling2D(pool_size= 2)(c3)
  c4 = Conv2D(filters= 128, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(m3)
  c4 = Dropout(rate= 0.2)(c4)
  c4 = Conv2D(filters= 128, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c4)

  # move into 5th layer : convolution 4 (last convolution step)
  m4 = MaxPooling2D(pool_size= 2)(c4)
  # deepest layer, shape = 8x8x256
  c5 = Conv2D(filters= 256, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(m4)
  c5 = Dropout(rate= 0.3)(c5)
  c5 = Conv2D(filters= 256, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c5)

  # move back into 4th layer : upconvolution 1
  u1 = Conv2DTranspose(filters= 128, kernel_size= 2, strides= 2, padding= 'same')(c5)
  u1 = concatenate([u1, c4])
  c6 = Conv2D(filters= 128, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(u1)
  c6 = Dropout(0.2)(c6)
  c6 = Conv2D(filters= 128, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c6)

  # move back into 3rd layer : upconvolution 2
  u2 = Conv2DTranspose(filters= 64, kernel_size= 2, strides= 2, padding= 'same')(c6)
  u2 = concatenate([u2, c3])
  c7 = Conv2D(filters= 64, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(u2)
  c7 = Dropout(0.2)(c7)
  c7 = Conv2D(filters= 64, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c7)

  # move back into 2nd layer : upconvolution 3
  u3 = Conv2DTranspose(filters= 32, kernel_size= 2, strides= 2, padding= 'same')(c7)
  u3 = concatenate([u3, c2])
  c8 = Conv2D(filters= 32, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(u3)
  c8 = Dropout(0.1)(c8)
  c8 = Conv2D(filters= 32, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c8)

  # move back into 1st layer : upconvolution 4
  u4 = Conv2DTranspose(filters= 16, kernel_size= 2, strides= 2, padding= 'same')(c8)
  u4 = concatenate([u4, c1], axis= 3)
  c9 = Conv2D(filters= 16, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(u4)
  c9 = Dropout(0.1)(c9)
  c9 = Conv2D(filters= 16, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c9)

  # return a classification model with sigmoid activation
  outputs = Conv2D(CLASSES, (1,1), activation= 'softmax')(c9)

  # define the beginning and starting points of the model aka. what you feed in and what you expect it to return
  model = Model(inputs = [inputs], outputs= [outputs])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  model.summary()

  return model


# convert to np arrays for the network
print('number of images omitted:', errors, '|| total images:', len(image_dataset) )

# label them as 0, 1, 2 because then you can use IoU
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
d, h, w = mask_dataset.shape
reshaped_masks = mask_dataset.reshape(-1,1)
print(reshaped_masks)
reshaped_masks_encoded = enc.fit_transform(reshaped_masks.ravel())
reshaped_masks_encoded_original_shape = reshaped_masks_encoded.reshape(d, h, w)
np.unique(reshaped_masks_encoded_original_shape)


# one hot encoding for the masks
CLASSES = 3
IMGH = 128
IMGW = 128

# this is the most simple way of encoding I found how to do it without reshaping arrays
# mask_dataset = tf.keras.utils.to_categorical(tf.convert_to_tensor(mask_dataset, dtype= tf.int64))

image_dataset = image_dataset.astype('float32')
mask_dataset = mask_dataset.astype('float32')
# mask_dataset = np.expand_dims(mask_dataset, axis= 3)
print('image_dataset:', image_dataset.shape, image_dataset.dtype)
print('mask_dataset:', mask_dataset.shape, mask_dataset.dtype)
# image_dataset = np.expand_dims(train_images, axis= 3)

# training set
train_frames = []
train_masks = []
# validation set
val_frames = []
val_masks = []
# test set
test_frames = []
test_masks = []

# split the datasets into test, validation, train
import pandas as pd

train_size, val_size, test_size = 0,0,0

def train_validate_test_split(train_percent=.75, validate_percent=.15, seed=None):


  train_end_index = int(len(image_dataset) * train_percent)
  val_end_index = int(len(image_dataset) * validate_percent) + train_end_index

  c = 0

  for i in range(train_end_index):
    train_frames.append(image_dataset[i])
    train_masks.append(mask_dataset[i])
    c+= 1

  train_size = c
  c = 0



  for i in range(train_end_index, val_end_index):
    val_frames.append(image_dataset[i])
    val_masks.append(mask_dataset[i])
    c += 1

  val_size = c
  c = 0

  for i in range(val_end_index, len(image_dataset)):
    test_frames.append(image_dataset[i])
    test_masks.append(mask_dataset[i])
    c += 1

  # print(c)

  test_size = c

# perform the split
train_validate_test_split()
print('total images:', len(image_dataset) + len(mask_dataset))
print('total train:', len(train_frames) + len(train_masks))
print('total val:', len(val_frames) + len(val_masks))
print('total test:', len(test_frames) + len(test_masks))

# class weights for even distribution
from sklearn.utils import class_weight

print(np.unique(reshaped_masks_encoded))

class_weights = class_weight.compute_class_weight('balanced', np.unique(reshaped_masks_encoded), reshaped_masks_encoded)
print(class_weights)

# apply to model
model = unet_model(128, 128, 3, 3)

train_frames = np.array(train_frames)
print('train_frames shape:', train_frames.shape)
train_masks = np.array(train_masks)
print('train_masks shape:', train_masks.shape)
val_frames = np.array(val_frames)
print('val_frames shape:', val_frames.shape)
val_masks = np.array(val_masks)
print('val_masks shape:', val_masks.shape)

from keras.utils import to_categorical

train_masks = train_masks - 1
val_masks = val_masks - 1
print(np.unique(train_masks), np.unique(val_masks))

train_masks = to_categorical(train_masks)
val_masks = to_categorical(val_masks)
print(train_masks.shape, val_masks.shape)

# this is the real shit
model.fit(train_frames,
          train_masks,
          batch_size= 16,
          verbose= 1, epochs= 25,
          validation_data= (val_frames, val_masks),
          # class_weight= class_weights,
          shuffle = False)

test_frames = np.array(test_frames)
test_masks = np.array(test_masks)
test_masks.shape
print(test_frames[0].shape)
result = model.predict(test_frames, batch_size=740)


o = Image.open('Terrain.jpg').convert('RGB')
o = o.resize((128,128))
o = asarray(o)
o = np.expand_dims(o, 0)
o.shape

model.predict(o, batch_size=1)


import json

print(type(model))
single = test_frames[0]
print(single.shape)
single = np.array(single)
single = np.expand_dims(single, 0)
print(single.shape)

single_result = model.predict(single, batch_size= 1)
single_result = tf.argmax(single_result, axis= -1)
single_result = np.squeeze(single_result, 0)
print(single_result.shape)
# plt.imshow(single_result)

single_result = single_result.astype('uint8')
single_result = np.array(single_result).tolist()
print(single_result)
single_result = json.dumps(single_result)
print(type(single_result))


new_img = Image.fromarray(np.array(json.loads(single_result), dtype= 'uint8'))
plt.imshow(new_img)

# there are 3 classes, and it did classifications on all 3 of them per pixel
pptime = tf.argmax(result[7], axis= -1)
plt.imshow(pptime)

import random

rows, cols = 5, 3

ref_test_images = test_frames.astype(int)
ref_test_masks = test_masks.astype(int)


fig, axs = plt.subplots(nrows= rows, ncols = cols, constrained_layout= True, figsize= (20,20), sharex= True, sharey= True)
titles = ['Input Image', 'Ground Truth', 'Predicted Mask']
r, c = 0, 0

while (r < rows):
  randomIndex = random.randint(0, 740)
  axs[r, c].imshow(ref_test_images[randomIndex])
  axs[r, c].get_xaxis().set_visible(False)
  axs[r, c].get_yaxis().set_visible(False)
  c += 1
  axs[r, c].imshow(ref_test_masks[randomIndex])
  axs[r, c].get_xaxis().set_visible(False)
  axs[r, c].get_yaxis().set_visible(False)
  c += 1
  axs[r, c].imshow(tf.argmax(result[randomIndex], axis= -1))
  axs[r, c].get_xaxis().set_visible(False)
  axs[r, c].get_yaxis().set_visible(False)
  c = 0
  r += 1

# fuck matplotlib this shit isnt working
# the axis titles go: Input Image, Ground Truth, Predicted Mask

# save the model into the TetraNet drive
random_name = str(random.random()).split('.')[1]
print('saved model: ', random_name + '.h5')

# uncomment if trying to save this model
# model.save(drive_dir + 'models/' + random_name + '.h5')
