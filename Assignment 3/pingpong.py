import cv2
import numpy as np
import math
import os
from objloader_simple import *


# ORB keypoint detector
orb = cv2.ORB_create()              
# create brute force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  

# Database of keypoints and descriptors for the markers, (marker_name, (kp, des)) pair
keyPointDataBase = {}


# Updates the database, MIN_MATCHES is 10 by default
def updateDataBase(markerName, MIN_MATCHES = 10):
  
  global keyPointDataBase

  markerModel = cv2.imread(markerName, 0)


  # Compute model keypoints and its descriptors
  kp_model, des_model = orb.detectAndCompute(markerModel, None) 

  # Update the keypoint and descriptor database
  keyPointDataBase[markerName] = (kp_model, des_model)


# Initialises the database of keypoints and descriptors of the markers
def initialiseDataBase(markerNameList, MIN_MATCHES = 10):
  
  for markerName in markerNameList:
    updateDataBase(markerName, MIN_MATCHES)


# Finds the midpoint of the given marker inside the video frame and returns it as numpy array, returna empty array if not enough matches
def findMarkerMidPointInFrame(frameKeyPointDescriptorTuple, markerModelShape, markerName, MIN_MATCHES = 10):
  
  midPointArray = np.array([])
  homography = None

  # Obtaining the keypoints and descriptors of the frame
  kp_frame, des_frame = frameKeyPointDescriptorTuple

  # Obtaining the keypoints and descriptors of the marker from the database
  kp_marker, des_marker = keyPointDataBase[markerName]

  # Finding the matches between the frame and the markerImage
  matches = bf.match(des_marker, des_frame)
  # Sorting out good matches
  matches = sorted(matches, key=lambda x: x.distance)
  # TODO: Hardcoded, need a better measure
  maxDistance = 30
  # Number of good matches
  good_matches = [m for m in matches if m.distance < maxDistance]

  #Checking for enough matches
  if len(good_matches) > MIN_MATCHES:
    # differentiate between source points and destination points
    src_pts = np.float32([kp_marker[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # compute Homography
    reprojThresh = 4.0
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reprojThresh)
    # draw matches.
    # frame = cv2.drawMatches(model, kp_model, frame, kp_frame,
    #                       matches[:MIN_MATCHES], 0, flags=2)

    if homography is not None:
      h, w = markerModelShape
      pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
      # project corners into frame
      dst = cv2.perspectiveTransform(pts, homography)
      # # connect them with lines  
      # frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 

      # Finding the midPoint of the four points obtained
      midPointArray = np.array( [ np.mean( dst[:, 0, 0] ), np.mean( dst[:, 0, 1] ) ] )
    else:
      print("Homography couldnt be computed for marker %s" % (markerName))

  else:
    # Not enough matches
    print ("Not enough matches for marker %s have been found - %d/%d" % ( markerName, len(good_matches),
                                                            MIN_MATCHES))

  return midPointArray


''' Returns a unit vector in the specified direction of the vector'''
def normalize(v):
  norm=np.linalg.norm(v, ord=2)
  if norm==0:
    return v
  return v/norm


def featureMatching(playerOneMarkerFileName, playerTwoMarkerFileName):
  ''' Match the features of model to scene '''

  # NOTE: Here, taking position of ball wrt Player 1 only

  # For player 1 marker model
  playerOneMarkermodel = cv2.imread(playerOneMarkerFileName, 0)
  playerOne_kp_model, playerOne_des_model = keyPointDataBase[playerOneMarkerFileName]

  # For destination marker model
  playerTwoMarkerModel = cv2.imread(playerTwoMarkerFileName, 0)
  playerTwo_kp_model, playerTwo_des_model = keyPointDataBase[playerTwoMarkerFileName]


  MIN_MATCHES = 10

  obj = OBJ('fox.obj', swapyz=True)
  intrinsicMatrix =  np.array([[872.5309487, 0.000000000000000000e+00, 418.59218933], 
  [0.000000000000000000e+00, 860.07596622, 244.32456827],
  [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

  cap = cv2.VideoCapture(0)

  # Setting the kinematic properties of the object
  objectOffset = (0,0,0)
  objectDistanceFromPlayerOne = 0.0
  objectSpeed = 0.5

  objectDirection = np.random.rand(2)
  # objectDirection = np.array([1, 0])
  objectDirection = normalize(objectDirection)

  object2DCoordinates = np.array([-1, -1])
  # Offset from Player 1
  object2DOffsetCoordinates = np.array([0, 0])


  playerOneScore = 0
  playerTwoScore = 0

  # Whose Hitting turn is it
  turn = "Player2"

  while True:

    objectSpeed += 0.05

    #Initialising homography as None
    homography = None

    # read the current frame
    ret, frame = cap.read()
    
    if not ret:
      print("Unable to capture video")
      return 

    # Getting the dimensions of the frame
    frameLength, frameWidth, frameDepth = frame.shape

    # Drawing the line, which denotes the net
    lineThickness = 2
    cv2.line(frame, (int(frameWidth/2), 0), (int(frameWidth/2), frameLength), (0,255,0), lineThickness)



    # Compute scene keypoints and its descriptors
    kp_frame, des_frame = orb.detectAndCompute(frame, None)
    # Match frame descriptors with model descriptors
    matches = bf.match(playerOne_des_model, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)
    # Sorting out good matches
    # TODO: Hardcoded, need a better measure
    maxDistance = 30
    # Number of good matches
    good_matches = [m for m in matches if m.distance < maxDistance]

    if len(good_matches) > MIN_MATCHES:
      # differentiate between source points and destination points
      src_pts = np.float32([playerOne_kp_model[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
      dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
      # compute Homography
      reprojThresh = 4.0
      homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reprojThresh)
      # draw matches.
      # frame = cv2.drawMatches(model, kp_model, frame, kp_frame,
      #                       matches[:MIN_MATCHES], 0, flags=2)
      
      # if a valid homography matrix was found render cube on model plane
      if homography is not None:
        h, w = playerOneMarkermodel.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, homography)
        # connect them with lines  
        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
        

        # Finding the midpoints for source and destination markers
        playerOneMarkerMidPointArray = np.array( [ np.mean( dst[:, 0, 0] ), np.mean( dst[:, 0, 1] ) ] )
        playerTwoMarkerMidPointArray = findMarkerMidPointInFrame((kp_frame, des_frame), playerTwoMarkerModel.shape, playerTwoMarkerFileName)


        # Set object 2D coordinates      
        if object2DCoordinates[0] == -1 and object2DCoordinates[1] == -1:
          object2DCoordinates = np.array(playerOneMarkerMidPointArray)
        else:
          object2DCoordinates = np.add(object2DCoordinates , objectDirection * np.float64(np.int64(objectSpeed))) # SO that the speed increases gradually
        
        # Updating the object offset coordinates
        object2DOffsetCoordinates = object2DCoordinates - playerOneMarkerMidPointArray

        objectOffset = ( (object2DOffsetCoordinates[0]), (object2DOffsetCoordinates[1]), objectOffset[2]) 
          
          
        # Updating the object distance from Player One
        vectorNorm = np.linalg.norm(object2DOffsetCoordinates, ord=2)
        objectDistanceFromPlayerOne = vectorNorm

        # Initialising the distance to infinity
        objectDistanceFromPlayerTwo = np.inf

        doesPlayerTwoExist = not (playerTwoMarkerMidPointArray.size == 0)

        
        if doesPlayerTwoExist:
          # Calculating object distance from destination
          vector = (playerTwoMarkerMidPointArray - object2DCoordinates)
          objectDistanceFromPlayerTwo = np.linalg.norm(vector, ord=2)

        ''' Checking for collision with marker which has its turn'''
        
        # ? Doubt if frame_height or frame_width
        objectInAreaOne = object2DCoordinates[0] < frameWidth/2
        objectInAreaTwo = object2DCoordinates[0] > frameWidth/2

        # If closer than this, then reflect
        objectDistanceThreshold = 50.0

        print(turn)
        print(objectInAreaOne)
        print(objectInAreaTwo)

        print(objectDistanceFromPlayerOne)
        print(objectDistanceFromPlayerTwo)

        # Reflecting code
        if turn == "Player1" and objectInAreaOne == True:
          # If less than threshold, stop moving
          if objectDistanceFromPlayerOne < objectDistanceThreshold:
            # Reversing the x direction
            objectDirection[0] = - objectDirection[0]
            turn = "Player2"

        if turn == "Player2" and objectInAreaTwo == True:
          # If less than threshold, stop moving
          if objectDistanceFromPlayerTwo < objectDistanceThreshold:
            # Reversing the x direction
            objectDirection[0] = - objectDirection[0]
            turn = "Player1"

        ''' Checking if ball is out of the playing area '''
        toRestart = 0

        # ? Doubt if height or length
        if object2DCoordinates[0] > frameWidth or object2DCoordinates[0] < 0:
          toRestart = 1
        
        if object2DCoordinates[1] > frameLength or object2DCoordinates[1] < 0:
          toRestart = 1

        if toRestart == 1:
          # Setting the kinematic properties of the object
          objectOffset = (0,0,0)
          objectDistanceFromPlayerOne = 0.0
          objectSpeed = 0.5

          objectDirection = np.random.rand(2)
          # objectDirection = np.array([1, 0])
          objectDirection = normalize(objectDirection)

          object2DCoordinates = np.array([-1, -1])
          # Offset from Player 1
          object2DOffsetCoordinates = np.array([0, 0])

          # Whose Hitting turn is it
          turn = "Player2"
          

          



        # obtain 3D projection matrix from homography matrix and camera parameters
        projection = projection_matrix(intrinsicMatrix, homography)  
        # project cube or model
        frame = render(frame, obj, projection, playerOneMarkermodel, objectOffset, False) 

          

        # if doesDirectionVectorExist and objectDistancefromDestination < objectDistancefromDestinationThreshold :
        #   objectOffset = ((objectOffset[0] + deltaDisplacementForObject[0]), (objectOffset[1] + deltaDisplacementForObject[1]), objectOffset[2])

      else:
        print("Homography couldnt be computed for marker %s" % (playerOneMarkerFileName))                    
      # show result
      # cv2.imshow('frame', frame)
      # cv2.waitKey(0)
    else:
      print ("Not enough matches for marker %s have been found - %d/%d" % (playerOneMarkerFileName, len(good_matches),
                                                          MIN_MATCHES))
    # show result
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break 

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

def render(img, obj, projection, model, positionOffset = (0, 0, 0), color=False):
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
      points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2] ] for p in points])

      positionOffset2D = positionOffset[:-1]
      dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
      positionOffsets =  [[ positionOffset2D], [ positionOffset2D], [ positionOffset2D] ] 
      dst = dst + np.array( positionOffsets )
      imgpts = np.int32(dst)
      if color is False:
          cv2.fillConvexPoly(img, imgpts, (137, 27, 211))


    return img

if __name__ == "__main__":
  
  # Selecting the source and destination markers to be used
  playerOneMarker = "marker_photos/marker_5.jpg"
  
  playerTwoMarker = "marker_photos/marker_4.jpg" 

  # Initialing the marker keypoint database with source and destination markers
  initialiseDataBase([playerOneMarker, playerTwoMarker])
  
  # Computes and renders the augmented image
  featureMatching(playerOneMarker, playerTwoMarker)