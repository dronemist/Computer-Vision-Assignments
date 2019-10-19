import cv2
import numpy as np
import math
import os
from objloader_simple import *

def featureMatching(fileName):
  ''' Match the features of model to scene '''
  model = cv2.imread(fileName, 0)
  # object to be projected
  obj = OBJ('fox.obj', swapyz=True)
  MIN_MATCHES = 10
  # ORB keypoint detector
  orb = cv2.ORB_create()              
  # create brute force matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
  # Compute model keypoints and its descriptors
  kp_model, des_model = orb.detectAndCompute(model, None)  
  intrinsicMatrix = np.load("calibration.npy")
  cap = cv2.VideoCapture(0)
  while True:
    # read the current frame
    ret, frame = cap.read()
    if not ret:
      print("Unable to capture video")
      return 
    # Compute scene keypoints and its descriptors
    kp_frame, des_frame = orb.detectAndCompute(frame, None)
    # Match frame descriptors with model descriptors
    matches = bf.match(des_model, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)
    # Sorting out good matches
    # TODO: Hardcoded, need a better measure
    maxDistance = 40
    # Number of good matches
    good_matches = [m for m in matches if m.distance < maxDistance]
    
    if len(good_matches) > MIN_MATCHES:
      # differenciate between source points and destination points
      src_pts = np.float32([kp_model[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
      dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
      # compute Homography
      reprojThresh = 4.0
      homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reprojThresh)
      # draw matches.
      frame = cv2.drawMatches(model, kp_model, frame, kp_frame,
                            matches[:MIN_MATCHES], 0, flags=2)
      # if a valid homography matrix was found render cube on model plane
      if homography is not None:
        # obtain 3D projection matrix from homography matrix and camera parameters
        projection = projection_matrix(intrinsicMatrix, homography)  
        # project cube or model
        frame = render(frame, obj, projection, model, False)                     
      # show result
      cv2.imshow('frame', frame)
      cv2.waitKey(0)
    else:
      print ("Not enough matches have been found - %d/%d" % (len(good_matches),
                                                            MIN_MATCHES))
    # show result
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break 
  cap.release()
  cv2.destroyAllWindows()
  return 0                                                        

def projection_matrix(camera_parameters, homography):
  """
  From the camera calibration matrix and the estimated homography
  compute the 3D projection matrix
  """
  # Compute rotation along the x and y axis as well as the translation
  homography = homography * (-1)
  rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
  col_1 = rot_and_transl[:, 0]
  col_2 = rot_and_transl[:, 1]
  col_3 = rot_and_transl[:, 2]
  # normalise vectors
  l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
  rot_1 = col_1 / l
  rot_2 = col_2 / l
  translation = col_3 / l
  # compute the orthonormal basis
  c = rot_1 + rot_2
  p = np.cross(rot_1, rot_2)
  d = np.cross(c, p)
  rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
  rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
  rot_3 = np.cross(rot_1, rot_2)
  # finally, compute the 3D projection matrix from the model to the current frame
  projection = np.stack((rot_1, rot_2, rot_3, translation)).T
  # printing (R, t)
  print(projection)
  return np.dot(camera_parameters, projection)

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 1
    h, w = model.shape

    for face in obj.faces:
      face_vertices = face[0]
      points = np.array([vertices[vertex - 1] for vertex in face_vertices])
      points = np.dot(points, scale_matrix)
      # render model in the middle of the reference surface. To do so,
      # model points must be displaced
      points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
      dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
      imgpts = np.int32(dst)
      if color is False:
          cv2.fillConvexPoly(img, imgpts, (137, 27, 211))

    return img

if __name__ == "__main__":
  featureMatching("marker01.png")