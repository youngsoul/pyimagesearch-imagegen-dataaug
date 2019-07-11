# set the matplotlib backend so figures an be saved in the background
import matplotlib
matplotlib.use('Agg')

# import the necessary packages
from pyimagesearch.resnet import ResNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-a", "--augment", type=int, default=-1, help="whether or not 'on the fly' data augmentatiopn should be used")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of epochs to train for
INIT_LR = 1e-1
BS = 8
EPOCHS = 50

# grab the list of images in our dataset directory, then initialize the list of data ( i.e. images ) and class images
logging.info("loading images...")
imagePaths = list(paths.list_images(args['dataset']))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename, load the image, and
    # resize it to be a fixed 64x64 pixels, ignoring aspect ratio
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64,64))

    # update the data and labels list,
    data.append(image)
    labels.append(label)

# convert the data into a numpy array, then preprocess it by scaling
# all pixel intensities to the range [0,1]
data = np.array(data, dtype="float")/255.0

logging.debug(f"data shape: {data.shape}")
logging.debug(f"Labels: {list(set(labels))}")

# encode the labels ( which are currently strings ) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, num_classes=2)

# partition the data into training and testing splits using 75%/25%
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# initialize the data augmenter as an 'empty' image data generator
# default constructor will perform no augmentation
aug =ImageDataGenerator()

# check to see if we are applying 'on the fly' data augmentation, and if so
# re-instantiate the object
if args['augment'] > 0:
    logging.debug("performing 'on the fly' data augmentation")
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

# initialize the optimizer and model
logging.debug("compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR/EPOCHS)
model = ResNet.build(64,64,3,2,(2,3,4),(32,64,128,256), reg=0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
logging.info(f"training network for {EPOCHS} epochs...")
H = model.fit_generator(
    aug.flow(trainX,trainY, batch_size=BS),
    validation_data=(testX,testY),
    steps_per_epoch=len(trainX)//BS,
    epochs=EPOCHS
)

# evaluate the network
logging.info("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
logging.info(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


