import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

# TODO: calculate feature points once to avoid recalculation each time
def imageMatcher(path, img):
  """
  Matches all the images in path folder and returns the most similar image 
  name. 
  """
  img = cv.imread('1/2.jpg', cv.IMREAD_GRAYSCALE)
  maxSimilarity = 0
  similarImage = '2.jpg'

  # feature detector
  featureDetector = cv.KAZE_create()

  # Keypoints and descriptor of main image
  keyPoints1, d1  = featureDetector.detectAndCompute(img, None)
  for fileName in os.listdir(path):

    # Filename shouldn't be name of this image
    if fileName != '2.jpg':
      print(fileName)
      imgToCompare = cv.imread(path + fileName, cv.IMREAD_GRAYSCALE)
      keyPoints2, d2  = featureDetector.detectAndCompute(imgToCompare, None)
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
if __name__ == '__main__':
  img = cv.imread('1/2.jpg', cv.IMREAD_GRAYSCALE)
  imageMatcher('1/', img)
