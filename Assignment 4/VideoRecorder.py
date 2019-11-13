import numpy as np
import cv2
import sys
from preprocess import preprocess

# variables
bgSubThreshold = 50
  # bool, whether the background captured

def createDatabase(path, sizeOfImage, imageName, numberOfImages, imagesToSkip):
  '''Adds numberOfImages images to path folder'''
  isBgCaptured = 0 
  cap = cv2.VideoCapture(0)
  count = numberOfImages
  while count > 0:
    # Reading frame by frame
    ret, frame = cap.read()
    cv2.imshow('frame1', frame)
    if isBgCaptured:
      # cv2.imshow('frame', frame)
      frame = cv2.resize(frame, sizeOfImage)
      if count % imagesToSkip == 0: # Images to skip denotes that we have to take every ith image
        cv2.imwrite(path + '/' + imageName + str(count) + '.jpg', frame)
      print(count)
      cv2.imshow('frame', frame)
      count -= 1 
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
      break
    elif k == ord('b'):
      # bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
      isBgCaptured = 1
      cv2.imwrite(path + '/' + imageName + str(count) + 'bg.jpg', frame)
      print('!!!Background Captured!!!')
  cap.release()
  cv2.destroyAllWindows()  

if __name__ == "__main__":
  createDatabase(sys.argv[1], (200, 200), sys.argv[2], 999, 10)
  