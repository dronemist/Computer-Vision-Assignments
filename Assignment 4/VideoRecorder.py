import numpy as np
import cv2
import sys

def createDatabase(path, sizeOfImage, imageName, numberOfImages):
  '''Adds numberOfImages images to path folder'''
  cap = cv2.VideoCapture(0)
  count = numberOfImages
  while count > 0:
    # Reading frame by frame
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    frame = cv2.resize(frame, sizeOfImage)
    cv2.imwrite(path + '/' + imageName + str(count) + '.jpg', frame)
    print(count)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count -= 1
  cap.release()
  cv2.destroyAllWindows()  

if __name__ == "__main__":
  createDatabase(sys.argv[1], (50, 50), sys.argv[2], 500)
  