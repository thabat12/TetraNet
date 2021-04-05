import cv2
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.models import load_model
from numpy import asarray
from PIL import Image, ImageOps
import azure_get_unet as azure_predict
# Since we are using the Azure API, there is not need to save the model to the local filesystem

# model = load_model("/static/model/trees-v1.h5")

# model prediction returns array of prediction
# input is a numpy array
def predict_frame(image):
  image = np.expand_dims(image, 0)
  result = model.predict(image, batch_size=1)
  result = np.squeeze(result, 0)
  result = tf.argmax(result, -1)
  return result

# for resizing the images after predicting the frames
def resize_frame(arr, shape):
  result = Image.fromarray(arr)
  result = result.resize(shape)
  result = asarray(result)
  return result

# change the alpha values of the segmentation masks for overlay
def convert_mask_alpha(image_arr):
  img_transparent = Image.fromarray(image_arr)
  imga = img_transparent.convert('RGBA')
  imga_data = imga.getdata()

  newData = []

  for item in imga_data:
    if (item[0] == 0):
      newData.append((0,0,0,0))
    else:
      # orange transparent mask
      newData.append((255,170,0,100))

  img_transparent.close()
  imga.putdata(newData)
  imga = np.array(imga)
  return imga


# generate the list for the segmentation frames based on video path
def get_segmentation_frames(video_path):
    # Step 1: create the cv2 video capture object
    vidObj = cv2.VideoCapture(video_path)
    # Step 2: capture the video frames and predict segmentation,
    #         then append the segmented frames
    mask_frames = []

    count = 0
    success = 1

    while (True):
        success, image = vidObj.read()
        if (success == 0):
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Using PIL to get the proper coloration from the cv2 capture
        image = Image.fromarray(image)

        # 128x128 grayscale for UNet model processing
        image = image.resize((128, 128))
        image = ImageOps.grayscale(image)
        image = asarray(image)
        # with the incoming frame, convert to numpy and uint8 dtype
        # and resize frames to 1080p values
        append = predict_frame(image)
        append = np.array(append)
        append = append.astype('uint8')
        append = resize_frame(append, (480, 270))

        # list 1920x1080p numpy arrays
        mask_frames.append(append)

    # Step 3: convert the lists to numpy, and cast into usable
    #         black/ white array data for the video writer
    mask_frames = np.array(mask_frames)
    mask_frames = mask_frames * 255
    # just a sanity check for the VideoWriter
    mask_frames = mask_frames.astype('uint8')
    # return usable arrays for video writing
    return mask_frames


# This function will overlay the mask frames with the original video frames
def get_segmentation_frames_compiled(video_path):
    # Step 1: retrieve the full sized segmentation frames
    print('Generating segmentation frames...')
    mask_frames_list = get_segmentation_frames(video_path)
    print('Segmentation frames finished')

    # Step 2: make a new cv2 video capture object for recycling the image files
    vidObj = cv2.VideoCapture(video_path)

    compiled_list = []

    frame = 0
    success = 1
    # per frame, compile the values
    while (True):
        success, image = vidObj.read()
        if (success == 0):
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((480, 270))
        image = image.convert('RGBA')
        image = np.array(image)
        mask = convert_mask_alpha(mask_frames_list[frame])
        add_imgs = cv2.addWeighted(image, 1.0, mask, 0.4, 0.0)
        add_imgs = Image.fromarray(add_imgs).convert('RGB')
        add_imgs = asarray(add_imgs)
        compiled_list.append(add_imgs)
        frame += 1
    # return the RGBA data list
    compiled_list = np.array(compiled_list)
    print('Frames are finished compiling')
    return compiled_list


# expects uint8, numpy preferrable
def frames_to_video(imput_list, name, isRGB):
    out = cv2.VideoWriter(name + '.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 24, (480, 270), isRGB)

    for i in range(len(imput_list)):
        out.write(imput_list[i])

    print('finished')

# input will be a PIL image
def overlay_mask_to_img(original_image):
  mask = original_image
  mask = mask.resize((128,128))
  mask = ImageOps.grayscale(mask)
  mask = asarray(mask)
  mask = predict_frame(mask)
  mask = np.array(mask)
  mask = mask.astype('uint8')
  mask = convert_mask_alpha(mask)
  mask = Image.fromarray(mask)
  mask = mask.resize((1200, 600))
  original_image = original_image.convert('RGBA')
  original_image = asarray(original_image)
  original_image = original_image.astype('uint8')
  mask = asarray(mask).astype('uint8')
  print(original_image.shape)
  add_imgs = cv2.addWeighted(original_image, 1.0, mask, 0.4, 0.0)
  return add_imgs