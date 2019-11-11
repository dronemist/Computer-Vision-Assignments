from keras.models import load_model
import argparse
import pickle
import cv2
import sys

from preprocess import preprocess

# Image to predict
cap = cv2.VideoCapture(0)
# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model("model")
lb = pickle.loads(open("label_bin", "rb").read())
while True:
  ret, image = cap.read()
  output = image.copy()

  image = preprocess(image)
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
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
