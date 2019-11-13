from Model import GestureRecognizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import preprocess
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

data = []
labels = []



baseFrameFileName = "./gestures/background/bg1.jpg"
backgroundModel = preprocess.getBaseBackGroundModel(baseFrameFileName)




# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('gestures')))
random.seed(42)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
	# load the image, resize it to 64x64 pixels (the required input
	# spatial dimensions of SmallVGGNet), and store the image in the
	# data list
	image = cv2.imread(imagePath)
	# image = cv2.resize(image, (50, 50))
	image = preprocess.preprocess(image)
	image = preprocess.removeBG(image, backgroundModel, learningRate = 0)
	image = preprocess.drawImageContours(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") #/ 255.0

labels = np.array(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

model = GestureRecognizer.build(width=50, height=50, depth=3,
	classes=len(lb.classes_))

# initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR = 0.05
EPOCHS = 10
BS = 32

# initialize the model and optimizer (you'll want to use
# binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=lb.classes_))	

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)

# Training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig('trainingLoss.jpg')

# Validation loss
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Validation Loss")
plt.legend()
plt.savefig('validationLoss.jpg')

# Saving the model
model.save("model")
f = open("label_bin", "wb")
f.write(pickle.dumps(lb))
f.close()
  