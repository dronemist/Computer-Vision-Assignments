import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
database = []
def imageMatcher(path, img):
  """
  Matches all the images in path folder and returns the most similar image 
  name. 
  """
  img = cv.imread('1/2.jpg', cv.IMREAD_GRAYSCALE)
  maxSimilarity = 0
  similarImage = '2.jpg'
  # Keypoints and descriptor of main image
  keyPoints1, d1  = database[int(similarImage[0]) - 1]
  for fileName in os.listdir(path):
    # Filename shouldn't be name of this image
    if fileName != '2.jpg':
      print(fileName)
      imgToCompare = cv.imread(path + fileName, cv.IMREAD_GRAYSCALE)
      keyPoints2, d2  = database[int(fileName[0]) - 1]
      print(len(keyPoints1))
      print(len(keyPoints2))
      index_params = dict(algorithm = 1, trees = 5)
      search_params = dict()
      # Matching the two images
      flann = cv.FlannBasedMatcher(index_params, search_params)
      matches = flann.knnMatch(np.asfarray(d1, np.float32), np.asfarray(d2, np.float32), k = 2)
      # NOTE: can be commented if image not being plotted
      matchesMask = [[0,0] for i in range(len(matches))]
      # Number of good matches
      good_points = []
      # Matching ratio between the two
      ratio = 0.8
      for i, (m, n) in enumerate(matches):
        if m.distance < ratio*n.distance:
            matchesMask[i]=[1,0]
            good_points.append(m)
      print(len(good_points))    
      # Max similarity  
      if(len(good_points) > maxSimilarity):
        maxSimilarity = len(good_points)
        similarImage = fileName
  print(similarImage)      
  # draw_params = dict(matchColor = (0,255,0),
  #                   singlePointColor = (255,0,0),
  #                   matchesMask = matchesMask,
  #                   flags = cv.DrawMatchesFlags_DEFAULT)      
  # similarImage = cv.drawMatchesKnn(img,keyPoints1,imgToCompare,keyPoints2,matches,None,**draw_params)
  # plt.imshow(similarImage),plt.show()   
  
def calculateDatatbase(path):
  global database
  database = [(None, None)] * len(os.listdir(path))
  # Number of features 
  numberOfFeatures = 5000
  # feature detector
  featureDetector = cv.ORB_create(nfeatures = numberOfFeatures)
  for fileName in os.listdir(path):
    # Reading image
    img = cv.imread(path + fileName, cv.IMREAD_GRAYSCALE)
    keyPoints, d  = featureDetector.detectAndCompute(img, None)
    database[int(fileName[0]) - 1] = (keyPoints, d)

if __name__ == '__main__':
  path = '1/'
  calculateDatatbase(path)
  img = cv.imread('1/2.jpg', cv.IMREAD_GRAYSCALE)
  imageMatcher('1/', img)
