# First, begin with the imports. We will primarily be using numpy and tensorflow/ keras
from PIL import Image, ImageOps
import numpy as np
from numpy import asarray
from matplotlib import pyplot as plt
from keras.utils import normalize
import os


# Our Google Drive is linked to the project's description. Simply pin that to your home for
# connecting to the datasets that are provided
from google.colab import drive
drive.mount('/content/drive') # run if not mounted already


# Check to ensure if GPU is available
%tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Define the dataset paths from the Drive
drive_dir = '/content/drive/MyDrive/Project TetraNet/files/dataset/'
image_names_dir = drive_dir + 'reference/'
mask_names_dir = drive_dir + 'masks/'

# Sort the file names... the sorted method should be enough to pair the images to the masks
image_names_list = sorted(os.listdir(image_names_dir))
mask_names_list = sorted(os.listdir(mask_names_dir))
print(image_names_list)
print(mask_names_list)

# This is the construction of the lists content for the image/mask datasets
image_dataset = []
mask_dataset = []
img_shape = (128,128)
for x,y in zip(image_names_list, mask_names_list):
  img = Image.open(image_names_dir + x)
  img = ImageOps.grayscale(img)
  img = img.resize(img_shape)
  img = asarray(img)
  image_dataset.append(img)
  mask = Image.open(mask_names_dir + y)
  mask = ImageOps.grayscale(mask)
  mask = mask.resize(img_shape)
  mask = asarray(mask)
  mask_dataset.append(mask)

# For storage optimization, cast the dataset variables as numpy arrays
image_dataset = np.array(image_dataset)
image_dataset = image_dataset.astype('uint8')
mask_dataset = np.array(mask_dataset)
mask_dataset = mask_dataset.astype('uint8')


"""
Project TetraNet uses datasets that are based off of our previous flight launches. To make
the best use of our data, we decided to utilize Keras libraries for image generators and
augmentation. So here is our implementation for the Keras datagens:

"""


# copy the image/mask datasets and increase dimensions for input in keras datagens
image_dataset_to_datagen = image_dataset.copy()
image_dataset_to_datagen = np.expand_dims(image_dataset_to_datagen, -1)

mask_dataset_to_datagen = mask_dataset.copy()
mask_dataset_to_datagen = np.expand_dims(mask_dataset_to_datagen, -1)

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# We will randomly rotate and zoom into the images. Since most of the data that the TetraNet
# devices will see will ultimately be different mutated shapes of trees/ vegetation, the
# transformations here are what we found to be the most reliable measure.
image_data_generator = ImageDataGenerator(
    featurewise_center= True,
    rotation_range= 90,
    zoom_range = [0.2, 1.0],
    dtype= 'float32'
)

image_datagen = image_data_generator
mask_datagen = image_data_generator

print(image_dataset.shape, mask_dataset.shape)

# the same seed will allow for the same transformations
seed = 1
# fitting will calculate any statistics required to actually perform the transformations
image_datagen.fit(image_dataset_to_datagen, augment= False, seed=seed)
mask_datagen.fit(mask_dataset_to_datagen, augment= False, seed=seed)

image_generator = image_datagen.flow(image_dataset_to_datagen, batch_size= 40, seed=seed, shuffle= False)
mask_generator = mask_datagen.flow(mask_dataset_to_datagen, batch_size=40, seed=seed, shuffle= False)

train_generator = zip(image_generator, mask_generator)

# This step is to visualize the data generated from the keras datagens
fig, axs = plt.subplots(nrows= 10, ncols = 2, constrained_layout= False, figsize= (100,100), sharex= True, sharey= True)
r, c = 0,0

train_X = image_generator.next()
train_X = train_X.astype('uint8')
train_y = mask_generator.next()
train_X = np.squeeze(train_X, -1)
train_y = np.squeeze(train_y, -1)




for r in range (10):
  axs[r,0].imshow(train_X[r])
  axs[r, 0].get_xaxis().set_visible(False)
  axs[r, 0].get_yaxis().set_visible(False)

  axs[r, 1].imshow(train_y[r])
  axs[r, 1].get_xaxis().set_visible(False)
  axs[r, 1].get_yaxis().set_visible(False)


# Concatenate all the randomly generated images into an appended dataset
# The seed values from the image and mask generators should return correlated images & masks

image_dataset_concat = image_generator.next()
mask_dataset_concat = mask_generator.next()

plt.imshow(np.squeeze(mask_dataset_concat[0], -1))

for i in range(10):
  image_dataset_concat = np.concatenate((image_dataset_concat, image_generator.next()))
  mask_dataset_concat = np.concatenate((mask_dataset_concat, mask_generator.next()))

print(image_dataset_concat.shape, mask_dataset_concat.shape)

# So after the keras datagen actions, the mask dataset values become interpolated and classes for [0,1]
# can become random as values such [0, 0.2231..., 0.474..., etc]. What we just did was simply redo all the
# values back to [0,1]
print(np.unique(mask_dataset_concat))
p = mask_dataset_concat.copy()
p = np.around(p, 0)
p = p.astype('uint8')
print(np.unique(p))

plt.imshow(np.squeeze(p[0], -1))

# This is the UNet model that we will be using
from vegetation_unet_model import unet_model

model= unet_model(img_shape[0], img_shape[1], 1, 2)

# Encode the mask data for proper size in the unet
from keras.utils import to_categorical
mask_dataset_concat = to_categorical(p)
print(p.shape, np.unique(p))

# Fit the model
model_history = model.fit(image_dataset_concat, mask_dataset_concat, validation_split= 0.1, epochs= 100, steps_per_epoch= 4, batch_size= 16, verbose= 1, shuffle= False)

# Evaluate the model's accuracy through the loss functions
"""
IMPORTANT: categorical_accuracy and loss functions are general terms to measure the model's categorical
           proper validation should also consider the intersection over union metric as well.
"""
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(100)

# Visualize the model's performance
plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


_, acc = model.evaluate(image_dataset_concat[0:30], mask_dataset_concat[0:30])


imgs = image_generator.next()
masks = mask_generator.next()
results = model.predict(imgs, batch_size= 40)

import random
random_name = 'trees-v1'
model.save(drive_dir + '/models/' + random_name + '.h5')

# Local file 7.png
new_img = Image.open(image_names_dir + '/7.png')
new_img = ImageOps.grayscale(new_img)
new_img = new_img.resize((128,128))
new_img = asarray(new_img)
save_img = Image.open(image_names_dir + '/7.png').convert('RGB')
save_img = asarray(save_img)
h,w,d = save_img.shape
new_img = asarray(new_img)
new_img = new_img.astype('uint8')
new_img = np.expand_dims(new_img, 0)

true_mask = Image.open('replace_img.png_annotation.ome.tiff')
true_mask = asarray(true_mask)

# This is how to extract the prediction masks from the UNet prediction
result = model.predict(new_img, batch_size= 1)
result = np.squeeze(result, 0)

from skimage.transform import resize
resized_img_mask = resize(result, (h,w))

list_images = []
list_images = np.array(list_images).astype(object)
print(list_images.dtype)



# A sample representation of the model's final input. This is the final result of the
# UNet's prediction. After this, the model is ready to be saved! 
titles = ['Input Image', 'True Mask', 'Predicted Mask']

for i in range(3):
  plt.figure(figsize=(h/12, w/12))
  plt.subplot(1, 3, i+1)
  plt.title(titles[i], fontsize = 25)
  if (i==0):
    plt.imshow(save_img)
  elif (i==1):
    plt.imshow(true_mask)
  else:
    plt.imshow(tf.argmax(resized_img_mask, -1))
  plt.axis('off')
  plt.show()
