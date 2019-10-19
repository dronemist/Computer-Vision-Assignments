import cv2
import numpy as np
import glob

# Calibrates the camera
def calibration():

  # termination criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  objp = np.zeros((6*7,3), np.float32)
  objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d point in real world space
  imgpoints = [] # 2d points in image plane. 
  cap = cv2.VideoCapture(0)
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  count = 50
  while count > 0:
    count -= 1
    print(count)
    ret, frame = cap.read()
    if not ret:
      print("Unable to capture video")
      return
    # img = cv2.imread(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(500)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break    
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
  mean_error = 0
  tot_error = 0
  
  # Calculating error
  for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

  # ? Current best is 0.1938  
  if(tot_error/len(objpoints) < 0.1938):  
    np.savetxt("calibration.txt", mtx)
  print("total error: " + str(tot_error/len(objpoints)))
  print(mtx)
  cap.release()  
        

cv2.destroyAllWindows()
if __name__ == "__main__":
  calibration()