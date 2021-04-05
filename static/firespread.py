import numpy as np
import imageio
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from numpy import asarray
from matplotlib import pyplot as plt
from keras.utils import normalize
import os
import random
import azure_get_unet
import random


# for testing purposes only
def img_dir_to_arr(image_dir):
    mask = azure_get_unet.get_mask(image_dir)
    mask = mask.astype('uint8')
    return mask


def generate_firespread_prediction(image_dir):
    original_shape = Image.open(image_dir).size

    result = img_dir_to_arr(image_dir)

    a = []
    for i in range(1, 100):
        a.append(random.uniform(0, 1))
    print(a)

    # Cell States
    # 0 = Clear, 1 = Fuel, 2 = Fire

    prob = 1.0  # probability of a cell being fuel, otherwise it's clear
    total_time = 300  # simulation time
    terrain_size = [128, 128]  # size of the simulation: 10000 cells

    result = asarray(result)

    result.flags
    state = result.copy()
    state.setflags(write=1)
    print(state[80][1])
    # states hold the state of each cell
    states = np.zeros((total_time, *terrain_size))
    states[0] = state
    states[0][1][110] = 2
    print(states.shape)
    print(states[0][1])

    z = np.where(states[0][1] == 1)
    print(z)

    # set the middle cell on fire!!!
    import random

    for t in range(1, total_time):
        # Make a copy of the original states
        states[t] = states[t - 1].copy()

        for x in range(1, terrain_size[0] - 1):
            for y in range(1, terrain_size[1] - 1):

                if states[t - 1, x, y] == 2:  # It's on fire
                    states[t, x, y] = 0  # Put it out and clear it

                    # If there's fuel surrounding it
                    # set it on fire!
                    temp = random.uniform(0, 1)
                    if states[t - 1, x + 1, y] == 1 and temp > prob:
                        states[t, x + 1, y] = 2
                    temp = random.uniform(0, 1)
                    if states[t - 1, x - 1, y] == 1 and temp > prob:
                        states[t, x - 1, y] = 2
                    temp = random.uniform(0, 1)
                    if states[t - 1, x, y + 1] == 1 and temp > prob:
                        states[t, x, y + 1] = 2
                    temp = random.uniform(0, 1)
                    if states[t - 1, x, y - 1] == 1 and temp > prob:
                        states[t, x, y - 1] = 2

    colored = np.zeros((total_time, *terrain_size, 3), dtype=np.uint8)

    # Color
    for t in range(states.shape[0]):
        for x in range(states[t].shape[0]):
            for y in range(states[t].shape[1]):
                value = states[t, x, y].copy()

                if value == 0:
                    colored[t, x, y] = [139, 69, 19]  # Clear
                elif value == 1:
                    colored[t, x, y] = [0, 255, 0]  # Fuel
                elif value == 2:
                    colored[t, x, y] = [255, 0, 0]  # Burning

    # Crop
    cropped = colored[:200, 1:terrain_size[0] - 1, 1:terrain_size[1] - 1]

    imageio.mimsave('./video.gif', cropped)

    resized_list = []

    for arr in cropped:
        img = Image.fromarray(arr)
        img = img.resize((original_shape[0], original_shape[1]))
        img = asarray(img)
        resized_list.append(img)

    resized_list = np.array(resized_list)
    print(resized_list.shape)

    imageio.mimsave('./ppea.gif', resized_list)
