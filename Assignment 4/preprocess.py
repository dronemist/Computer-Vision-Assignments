import cv2
import numpy as np
from imutils import paths

# ''' Preprocesses data vector of images'''
# def preprocess(data): 
#   # fgbg = cv2.createBackgroundSubtractorMOG2(0, 50)
#   count = 1
#   for image in data:
#     # fgmask = fgbg.apply(image,learningRate=0)
#     # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     # # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#     # kernel = np.ones((3, 3), np.uint8)
#     # fgmask = cv2.erode(fgmask, kernel, iterations=1)
#     # res = cv2.bitwise_and(image, image, mask=fgmask)
#     gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (41, 41), 0)
#     ret, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
#     cv2.imshow('asd', thresh)
#     cv2.waitKey(0)
#     # cv2.imwrite('preprocessed/image' + str(count) + 'jpg', fgmask) 
#     count += 1
    
''' For histogram equalization of the image '''
def histogramEqualizeImage(image):
  img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

  # equalize the histogram of the Y channel
  img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

  # convert the YUV image back to RGB format
  img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
  # img_output = cv2.resize(img_output, (int(img_output.shape[1]/5), int(img_output.shape[0]/5)))
  return img_output

''' Preprocesses the image '''
def preprocess(image):
  grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  resizedImage = cv2.resize(grayImage, (50, 50))
  # equalizedImage = histogramEqualizeImage(resizedImage)
  equalizedImage = cv2.equalizeHist(resizedImage)
  # scale the pixel values to [0, 1]
  scaledImage = equalizedImage.astype("float") / 255.0

  outputImage = np.empty((scaledImage.shape[0], scaledImage.shape[1], 3))

  # Made Grayscale 3 times for easier code change and averaging out of the weights
  outputImage[:,:,0] = scaledImage
  outputImage[:,:,1] = scaledImage
  outputImage[:,:,2] = scaledImage

  return outputImage

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