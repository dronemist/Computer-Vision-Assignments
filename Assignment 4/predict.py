from keras.models import load_model
import argparse
import pickle
import cv2
import sys

from preprocess import *

bgSubThreshold = 50
isBgCaptured = 0
bgModel = None

# Image to predict
cap = cv2.VideoCapture(0)
# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model("model2")
lb = pickle.loads(open("label_bin2", "rb").read())


while True:
  ret, image = cap.read()
  output = image.copy()
  output = cv2.resize(output, (200, 200))
  cv2.imshow("Imageq", output)
  if isBgCaptured == 1:  # this part wont run until background captured
    image = removeBG(image, bgModel, 0)
    image = drawImageContours(image)
    image = preprocess(image)
    # output = image.copy()
    # image = cv2.resize(image, (50, 50))

    # # scale the pixel values to [0, 1]
    # image = image.astype("float") / 255.0

    image = image.reshape((1, image.shape[0], image.shape[1],
      image.shape[2]))

    # make a prediction on the image
    preds = model.predict(image)

    # find the class label index with the largest corresponding
    # probability
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]

    # draw the class label + probability on the output image
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
      (0, 0, 255), 2)
    # show the output image  
    cv2.imshow("Image", output)
  k = cv2.waitKey(10)
  if k == 27:
    break
  elif k == ord('b'):
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    isBgCaptured = 1

cap.release()
cv2.destroyAllWindows()
