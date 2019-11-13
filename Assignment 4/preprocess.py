import cv2
import numpy as np
import copy
import os
from imutils import paths

''' For histogram equalization of the image '''
def histogramEqualizeImage(image):
  img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

  # equalize the histogram of the Y channel
  img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

  # convert the YUV image back to RGB format
  img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
  # img_output = cv2.resize(img_output, (int(img_output.shape[1]/5), int(img_output.shape[0]/5)))
  return img_output

''' Given an image, draw the contours in it after binary thresholding '''
def drawImageContours(img):

  threshold = 60  #  BINARY threshold
  blurValue = 5 # GaussianBlur parameter

  # convert the image into binary image
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
  # cv2.imshow('blur', blur)
  ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
  # cv2.imshow('ori', thresh)
  drawing = thresh
  # drawing = np.zeros(img.shape, np.uint8)

  # # get the coutours
  # thresh1 = copy.deepcopy(thresh)
  # _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # length = len(contours)
  # maxArea = -1
  # if length > 0:
  #   for i in range(length):  # find the biggest contour (according to area)
  #     temp = contours[i]
  #     area = cv2.contourArea(temp)
  #     if area > maxArea:
  #       maxArea = area
  #       ci = i

  #   res = contours[ci]
  #   hull = cv2.convexHull(res)
  #   cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
  #   cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

  return drawing


'''' Givn a base file, create its background model '''
def getBaseBackGroundModel(baseFrameFileName):
  learningRate = 1
  #Learning the base background 
  bgSubThreshold = 50
  backgroundModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
  baseFrame = cv2.imread(baseFrameFileName)
  baseFrame = cv2.resize(baseFrame, (50, 50))
  # baseFrame = histogramEqualizeImage(baseFrame)
  backgroundModel.apply(baseFrame,learningRate=learningRate)
  return backgroundModel


''' Removes the background from the frame '''
def removeBG(frame, backgroundModel, learningRate):
  learningRate = 0
  fgmask = backgroundModel.apply(frame,learningRate=learningRate)
  # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

  kernel = np.ones((3, 3), np.uint8)
  fgmask = cv2.erode(fgmask, kernel, iterations=1)
  res = cv2.bitwise_and(frame, frame, mask=fgmask)
  return res

baseFrameFileName = "./gestures/background/bg" + str(115) + ".jpg"
backgroundModel = getBaseBackGroundModel(baseFrameFileName)

''' Preprocesses the image '''
def preprocess(image):
  # grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # resizedImage = cv2.resize(grayImage, (50, 50))
  # # equalizedImage = histogramEqualizeImage(resizedImage)
  # equalizedImage = cv2.equalizeHist(resizedImage)
  # # scale the pixel values to [0, 1]
  # # equalizedImage = fgbg.apply(equalizedImage,learningRate=0.1)
  # scaledImage = equalizedImage.astype("float") / 255.0


  # # Made Grayscale 3 times for easier code change and averaging out of the weights
  
  # image = removeBG(image, backgroundModel, learningRate = 0)
  # image = drawImageContours(image)
  outputImage = cv2.resize(image, (50, 50))
  # outputImage = outputImage.astype("float") / 255.0
  image = np.empty((outputImage.shape[0], outputImage.shape[1], 3))
  image[:,:,0] = outputImage
  image[:,:,1] = outputImage
  image[:,:,2] = outputImage
  return image

if __name__ == "__main__":

  # for i in range(1,500):
    # baseFrameFileName = "./gestures/background/bg" + str(115) + ".jpg"
    baseFrameFileName = "./gesturesTemp2/previous/previous999bg.jpg"
    backgroundModel = getBaseBackGroundModel(baseFrameFileName)
    data = []
    # print(i)
    imagePaths = sorted(list(paths.list_images('gesturesTemp2/previous')))
    for imagePath in imagePaths:
      # load the image, resize it to 50x50 pixels 
      # , and store the image in the data list
      image = cv2.imread(imagePath)
      # image = histogramEqualizeImage(image)
      image = cv2.resize(image, (50,50))
      image = removeBG(image, backgroundModel, learningRate = 0)
      image = drawImageContours(image)
      label = imagePath.split(os.path.sep)[-1]
      print(label)
      cv2.imwrite('preprocessed/previous/' + str(label), image)
      # cv2.imshow('da', image)
      cv2.waitKey(0)
      
      # edges = cv2.Canny(image,10,200) 
      # cv2.waitKey(0)
      # Display edges in a frame 
      # cv2.imshow('Edges',edges)