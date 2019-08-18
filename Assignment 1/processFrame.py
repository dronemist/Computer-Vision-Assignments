# Program to extract frames
import cv2
import numpy as np

# Function to identify end points
def EndPoints(cannyImage, rho, theta):
    # rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)

    # Number of points to be checked
    limit = 50

    # Minimum number of votes
    threshold = 30

    # Maximum possible length
    maxLength = 100

    # Number of times loop is run
    frequency = 10
    # Main loop
    cnt = limit
    cnt2 = 0
    currentR = maxLength
    previousR = 0
    while frequency != 0:
        while cnt != 0:
            x1 = int(a * rho - currentR * (-b))
            y1 = int(b * rho - currentR * (a))
            if cannyImage[x1][y1] != 0:
                cnt2 += 1
            if cnt2 > threshold:
                temp = previousR
                previousR = currentR
                currentR = int(temp + previousR)/2
                break
            cnt -= 1
            currentR += 1
        cnt = limit
        cnt2 = 0  
        frequency -= 1
    return currentR    

# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0

    # background subtractor
    backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()

    # background subtractor MOG
    bs2 = cv2.bgsegm.createBackgroundSubtractorMOG()
    # cleaning image kernel

    kernel = np.ones((3,3), np.uint8)

    # making the background subtractor learn
    # vidObjLearn = cv2.VideoCapture(path)
    # countLearn = 0	
    # successLearn, imageLearn = vidObjLearn.read()
    # while successLearn:

    #     countLearn += 1

    #     successLearn, imageLearn = vidObjLearn.read()
    #     foregroundLearn = backgroundSubtractor.apply(imageLearn, learningRate = 0.01)
	
    # checks whether frames were extracted 
    success, image = vidObj.read() 

    # Video writer
    # foreground = backgroundSubtractor.apply(image)
    height, width, layers = image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('output.mp4', fourcc, 30, (width, height))

    # Processing the image
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        
        # foreground
        foreground = bs2.apply(image)
        if foreground is None:
            break
        
        # Displaying the part in focus
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 300, 300)
        # cv2.imshow('image', foreground)
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        #     break

        # Processing the image of clean it
        cleanedImage = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        
        # Canny image detector
        # Modify lower and upper threshold for better edges
        cannyImage = cv2.Canny(cleanedImage, 10, 90, apertureSize = 3)
        
        # Hough lines 
        lines = cv2.HoughLines(cannyImage, 1, np.pi/180, 120)
        temp = 0

        # r = EndPoints(cannyImage, lines[0])
        
        # Taking the average of the lines
        if np.shape(lines) == ():
            numLines = 0
        else:
            numLines = (np.shape(lines))[0]
        x1Sum = 0
        y1Sum = 0
        x2Sum = 0
        y2Sum = 0		
        if lines is not None:
            # Applying hough transform
            # r = 875
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    r = EndPoints(cannyImage, rho, theta)
                    print(r)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + r*(-b))
                    y1 = int(y0 + r*(a))
                    x2 = int(x0 - r*(-b))
                    y2 = int(y0 - r*(a))
                    temp = temp + 1
                    x1Sum = x1Sum + x1
                    x2Sum = x2Sum + x2
                    y1Sum = y1Sum + y1
                    y2Sum = y2Sum + y2
        if numLines != 0:		
            cv2.line(image, (int(x1Sum/numLines), int(y1Sum/numLines)), (int(x2Sum/numLines), int(y2Sum/numLines)), (255, 0, 0), 2)

        # Converting it back to video
        # Uncomment this line to write to video
        # video.write(image)

    video.release()
    vidObj.release()
    vidObjLearn.release()
    cv2.destroyAllWindows()
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("Videos/1.mp4")