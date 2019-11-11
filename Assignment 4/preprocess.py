import cv2
import numpy as np
from imutils import paths

''' Preprocesses data vector of images'''
def preprocess(data): 
  fgbg = cv2.createBackgroundSubtractorMOG2(0, 50)
  count = 1
  for image in data:
    fgmask = fgbg.apply(image,learningRate=0)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(image, image, mask=fgmask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (41, 41), 0)
    ret, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    cv2.imshow('asd', thresh)
    cv2.waitKey(0)
    # cv2.imwrite('preprocessed/image' + str(count) + 'jpg', fgmask) 
    count += 1
    

if __name__ == "__main__":
  data = []
  imagePaths = sorted(list(paths.list_images('gestures')))
  for imagePath in imagePaths:
    # load the image, resize it to 50x50 pixels 
    # , and store the image in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (50, 50))
    data.append(image)
  preprocess(data)