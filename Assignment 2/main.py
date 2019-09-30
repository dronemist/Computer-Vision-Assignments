import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt
import os
#Databse of image names with their keypoints and descriptors
database = {}
featureDetector = cv.xfeatures2d.SIFT_create()
#Dictionary of image name and a bool to find whether image is already stitched or not
# imagesAlreadyStitched = {}

def selectBaseImage(fileNameList, path):
  votingTable = {}
  answer = ""
  for file in fileNameList:
    votingTable[file] = 0
  for file in fileNameList:
    tempList = fileNameList  
    currentHomographyTable = [] 
    currentImageName = file
    # Calculating pairwise homography for current image
    sum = np.zeros((3,3))  
    while len(tempList) != 0:
      (Homography, status, similarImageName) = imageMatcher(path, currentImageName, tempList)
      sum += Homography
      currentHomographyTable.append((similarImageName, Homography))
      tempList = list(image for image in tempList if not(image == similarImageName))
    average = sum / len(fileNameList)
    minImageName = currentImageName
    minNorm = np.inf
    for (similarImageName, Homography) in currentHomographyTable:
      currentNorm = np.linalg.norm(Homography - average, 2)
      if(currentNorm < minNorm):
        minImageName = similarImageName
        minNorm = currentNorm
    votingTable[minImageName] += 1
  maxVotes = 0  
  for file in fileNameList:
    if votingTable[file] > maxVotes:
      answer = file
      maxVotes = votingTable[file]
  return answer

def readEqualisedImage(path):
  img = cv.imread(path)
  img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

  # equalize the histogram of the Y channel
  img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

  # convert the YUV image back to RGB format
  img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
  img_output = cv.resize(img_output, (int(img_output.shape[1]/5), int(img_output.shape[0]/5)))
  return img_output

#To trim the black protion of the image
def trimImage(image):

    #crop top
    while np.sum(image[0]) == 0:
      image = image[1:]
    #crop bottom
    while np.sum(image[-1]) == 0:
      image = image[:-2]
    #crop left
    while np.sum(image[:, 0]) == 0:
      image = image[:, 1:]
    #crop right
    while np.sum(image[:, -1]) == 0:
      image = image[:, :-2]

    return image

def blend(result, startX, startY, endX, endY, img):
  weights = (np.ones((endY + 2, endX + 2, 2))) * np.inf
  resultIsLeft = False
  # 1 is new image and 0 is result image
  # distance from left border
  for row in range(startY, endY):
    minCol = np.inf
    for column in range(startX, endX):
      if result[row, column].any() != 0:
        if(minCol == np.inf):
          minCol = column  
        if minCol == startX:
          weights[row, column, 1] = (column - minCol)
        else:
          weights[row, column, 0] = (column - minCol) 

  # distance from right border
  for row in range(startY, endY):
    maxCol = 0
    for column in reversed(range(startX, endX)):
      if result[row, column].any() != 0:
        if(maxCol == 0):
          maxCol = column
        if(maxCol == endX - 1):  
          weights[row, column, 1] = min(weights[row, column, 1], (maxCol - column))
        else:
          weights[row, column, 0] = min(weights[row, column, 0], (maxCol - column))  

  # distance from top border
  for column in range(startX, endX):
    minRow = np.inf
    for row in range(startY, endY):
      if result[row, column].any() != 0:
        if minRow == np.inf:
          minRow = row  
        if minRow == startY:
          weights[row, column, 1] = min(weights[row, column, 1], row - minRow)
        else:
          weights[row, column, 0] = min(weights[row, column, 0], row - minRow)

  # distance from bottom border
  for column in range(startX, endX):
    maxRow = 0
    for row in reversed(range(startY, endY)):
      if result[row, column].any() != 0:
        if maxRow == 0:
          maxRow = row 
        if maxRow == endY - 1:
          weights[row, column, 1] = min(weights[row, column, 1], maxRow - row)
        else:
          weights[row, column, 0] = min(weights[row, column, 0], maxRow - row)

  for row in range(startY, endY):
      for column in range(startX, endX):
        if result[row, column].any() != 0:
          # weight1 = 0.5
          weight1 = weights[row, column, 0]
          pixel1 = result[row, column]
          # weight2 = 0.5
          weight2 = weights[row, column, 1]
          pixel2 = img[row - startY, column - startX]
          sum = weight1 + weight2
          weight1 = (weight1)/ (sum)
          weight2 = (weight2)/ (sum)
          normalisedValue = cv.addWeighted(pixel1, weight1, pixel2, weight2, 0)
          result[row, column][0] = normalisedValue[0][0]
          result[row, column][1] = normalisedValue[1][0]
          result[row, column][2] = normalisedValue[2][0]
        else:
          result[row, column] = img[row - startY, column - startX]
  return result                             

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    print((h2, w2))
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
    ratio = 0.7
    pts2_ = cv.perspectiveTransform(pts2,(H * ratio + identity * (1 - ratio)))
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    print(pts2_)
    result = cv.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))

    startX = t[0]
    startY = t[1]
    endX = w1+t[0]
    endY = h1+t[1]
    result = blend(result, startX, startY, endX, endY, img1)


    # result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    # alpha = 0.5
    # beta = (1.0 - alpha)
    # result = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    return result


def imageMatcher(path, imgName, imagesFromWhichToSelect):
  """
  Matches all the images in path folder and returns the most similar image, also returns the homography with the best image
  name. 
  """

  maxSimilarity = 0
  similarImageName = '2.jpg'
  # Keypoints and descriptor of main image
  keyPoints1, d1  = database[imgName]
  best_good_points = []
  for fileName in imagesFromWhichToSelect:
    # Filename shouldn't be name of this image

    # print(fileName)
    keyPoints2, d2  = database[(fileName)]
    # print(len(keyPoints1))
    # print(len(keyPoints2))
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
    # print(len(good_points))
    # Max similarity
    if(len(good_points) > maxSimilarity):
      maxSimilarity = len(good_points)
      similarImageName = fileName
      best_good_points = list(good_points)
  # print(similarImageName)
  similarImage = readEqualisedImage(path + similarImageName)
  # Estimating Homography
  obj = np.empty((len(best_good_points), 2), dtype=np.float32)
  scene = np.empty((len(best_good_points), 2), dtype=np.float32)

  keyPoints2 = (database[similarImageName])[0]
  for i in range(len(best_good_points)):
    # -- Get the keypoints from the good matches
    obj[i, 0] = keyPoints1[best_good_points[i].queryIdx].pt[0]
    obj[i, 1] = keyPoints1[best_good_points[i].queryIdx].pt[1]
    scene[i, 0] = keyPoints2[best_good_points[i].trainIdx].pt[0]
    scene[i, 1] = keyPoints2[best_good_points[i].trainIdx].pt[1]

  # draw_params = dict(matchColor = (0,255,0),
  #                  singlePointColor = (255,0,0),
  #                  matchesMask = matchesMask,
  #                  flags = cv.DrawMatchesFlags_DEFAULT)
  # matchingImage = cv.drawMatchesKnn(similarImage,keyPoints2,img,keyPoints1,matches,None,**draw_params)
  # plt.imshow(matchingImage),plt.show()

  reprojThresh = 4.0
  (Homography, status) = cv.findHomography(scene, obj, cv.RANSAC, reprojThresh)
  return (Homography, status, similarImageName)

def updateDatabase(image, imageName):
  global database
  global featureDetector

  keyPoints, d = featureDetector.detectAndCompute(image, None)
  database[imageName] = (keyPoints, d)

def calculateDatabase(path):
  global database
  global featureDetector

  for fileName in os.listdir(path):
    # Reading image
    img = readEqualisedImage(path + fileName)
    print(img.shape)
    keyPoints, d  = featureDetector.detectAndCompute(img, None)
    database[fileName] = (keyPoints, d)

if __name__ == '__main__':
  #TODO: Resize images
  path = sys.argv[1]
  calculateDatabase(path)
  fileNameList = os.listdir(path)
  fileNameList = sorted(fileNameList)
  # baseFileName = selectBaseImage(fileNameList, path)
  baseFileName = '2.jpg'
  print(baseFileName)
  imagesRemainingToBeStitched = list(image for image in fileNameList if not(image == baseFileName))
  # Will keep modifying this baseImage by stitching images to it one by one
  baseImg = readEqualisedImage(path + baseFileName)
  baseImgName = "Base"
  updateDatabase(baseImg, baseImgName)

  while not(len(imagesRemainingToBeStitched) == 0): #Till no remaining images
    (Homography, status, similarImageName) = imageMatcher(path, baseImgName, imagesRemainingToBeStitched)
    # Removing the similar image from imagesRemainingToBeStitched
    print(similarImageName)
    print(Homography)
    len_temp = len(imagesRemainingToBeStitched)
    imagesRemainingToBeStitched = list(image for image in imagesRemainingToBeStitched if not(image == similarImageName))
    if(len(imagesRemainingToBeStitched) == len_temp):
      break
    similarImage = readEqualisedImage(path + similarImageName)
    if baseImg is None:
      print("Yes")
    else:
      print("No")

    result = warpTwoImages(baseImg, similarImage, Homography)
    # result = cv.warpPerspective(similarImage, Homography, (similarImage.shape[1] + baseImg.shape[1], similarImage.shape[0]))
    # result[0:baseImg.shape[0], 0:baseImg.shape[1]] = baseImg
    # result = trimImage(result)
    baseImg = result
    # baseImg = cv.resize(result, (similarImage.shape[1], similarImage.shape[0]))
    baseImg1 = cv.resize(result, (int(baseImg.shape[1] / 5), int(baseImg.shape[0] / 5)))
    if(len(imagesRemainingToBeStitched) != 0):
      updateDatabase(baseImg, baseImgName)
    cv.imshow("Result", baseImg1)
    cv.waitKey(0)
    print(baseImg.shape)
    print(similarImage.shape)

  baseImg = cv.resize(baseImg, (int(baseImg.shape[1] / 5), int(baseImg.shape[0] / 5)))
  cv.imwrite("Result.jpg", baseImg)
