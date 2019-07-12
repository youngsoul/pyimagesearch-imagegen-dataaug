# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-o", "--output", required=True, help="path to the output directory to store augmentations examples")
ap.add_argument("-t", "--total", type=int, default=100, help="# of training samples to generate")
ap.add_argument("-p", "--prefix", type=str, default="image", help="prefix to add to all generated images")

args = vars(ap.parse_args())

# load the input image, convert itto a numpy array and then reshape it to have an extra dimension
logging.debug(f"loading example image: {args['image']}")
image = load_img(args['image'])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# construct the image generator for the data augmentation then
# initialize the total number of images generated thus far
aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
total = 0

# construct the actual python generator
logging.debug("generating images...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=args['output'], save_prefix=args['prefix'], save_format="jpg")

for image in imageGen:
    # increment counter
    total += 1

    # if we have reach the specifiedc number of examples, break from the loop
    if total == args['total']:
        break
