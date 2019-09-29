import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
#Databse of image names with their keypoints and descriptors
database = {}
featureDetector = cv.xfeatures2d.SIFT_create()
#Dictionary of image name and a bool to find whether image is already stitched or not
# imagesAlreadyStitched = {}

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

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    print((h2, w2))
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    print(pts2_)
    result = cv.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    # alpha = 0.5
    # beta = (1.0 - alpha)
    # result = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    return result


def imageMatcher(path, img, imgName, imagesFromWhichToSelect):
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
  print(similarImageName)
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
  path = '1/'
  calculateDatabase(path)
  fileNameList = os.listdir(path)
  fileNameList = sorted(fileNameList)
  baseFileName = fileNameList[0]
  imagesRemainingToBeStitched = list(fileNameList[1:])
  #Will keep modifying this baseImage by stitching images to it one by one
  baseImg = readEqualisedImage(path + baseFileName)
  baseImgName = "Base"
  updateDatabase(baseImg, baseImgName)

  while not(len(imagesRemainingToBeStitched) == 0): #Till no remaining images
    (Homography, status, similarImageName) = imageMatcher(path, baseImg, baseImgName, imagesRemainingToBeStitched)
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
    result = trimImage(result)
    baseImg = result
    # baseImg = cv.resize(result, (similarImage.shape[1], similarImage.shape[0]))
    baseImg1 = cv.resize(result, (416, 234))
    if(len(imagesRemainingToBeStitched) != 0):
      updateDatabase(baseImg, baseImgName)
    cv.imshow("Result", baseImg1)
    cv.waitKey(0)
    print(baseImg.shape)
    print(similarImage.shape)

  baseImg = cv.resize(baseImg, (416, 234))
  cv.imwrite("Result.jpg", baseImg)
