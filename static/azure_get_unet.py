import requests
from PIL import Image
import numpy as np


def get_mask(image_dir,testing=False):
    # Here, we will access the Azure API and get the numpy mask for the prediction
    image_dir_local = ''
    if (testing):
        image_dir_local = 'uploads/images/test.png'
    else:
        image_dir_local = image_dir

    print(image_dir_local, 'is image dir')
    img = Image.open(image_dir_local)
    files = {'image': open(image_dir_local, 'rb').read()}
    print(files)
    response = requests.post('http://67515655-f00a-44a0-a447-22a76351d991.eastus.azurecontainer.io/score',
                             files=files)

    new_img = response.json()
    new_img = np.array(new_img, dtype='uint8')
    new_img = new_img * 255

    return new_img


def save_image(arr, name='default'):
    # Save the file to the uploads folder for future use
    new_img = Image.fromarray(arr)
    saved_file_dir = 'uploads/images/' + name + '.png'
    new_img.save(saved_file_dir)

    return saved_file_dir