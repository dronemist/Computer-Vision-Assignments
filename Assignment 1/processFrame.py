# Program to extract frames
import cv2
import numpy as np

def getMidPointAndLength(cannyImage):

    '''This function returns the mid point and length of the instrument'''

    numLines = 0
    angleSum = 0
    midPointX = 0
    midPointY = 0
    x1Min = 1e6
    x2Max = -1e6
    y1Min = 1e6
    y2Max = -1e6
    minLineLength = 50
    maxLineGap = 10
    lines = cv2.HoughLinesP(cannyImage, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    x1Sum = 0
    y1Sum = 0
    x2Sum = 0
    y2Sum = 0
    temp = 0
    try:
        if (lines[0] is not None):
            for line in lines:
                x1 = line[0,0]
                y1 = line[0,1]
                x2 = line[0,2]
                y2 = line[0,3]
                if y1 < y1Min:
                    x1Min = x1
                    y1Min = y1
                if y2 > y2Max:
                    x2Max = x2
                    y2Max = y2
                if x2-x1 == 0:
                    angleSum = angleSum - np.arctan(np.inf)
                else:
                    angleSum = angleSum - np.arctan((y2-y1)/(x2-x1))
                x1Sum = x1Sum + x1
                y1Sum = y1Sum + y1
                x2Sum = x2Sum + x2
                y2Sum = y2Sum + y2
                numLines = numLines + 1
    except:
        # do nothing
        temp += 1

    if numLines != 0:
        midPointX = (x1Sum + x2Sum) / (2 * numLines)
        midPointY = (y1Sum + y2Sum) / (2 * numLines)
    lineLength = np.sqrt((x1Min - x2Max) * (x1Min - x2Max) + (y1Min - y2Max) * (y1Min - y2Max))
    return (midPointX, midPointY, lineLength)

def drawLines(cannyImage, image, midPointX, midPointY, lineLength):

    '''This function draws the line on the image'''

    # Taking the average of the lines
    x1Sum = 0
    y1Sum = 0
    x2Sum = 0
    y2Sum = 0
    numLines = 0
    normalHoughLines = cv2.HoughLines(cannyImage, 1, np.pi / 180, 150)
    if normalHoughLines is not None:
        # Applying hough transform
        r = lineLength / 2 + 50
        for line in normalHoughLines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                # x0 = a*rho
                # y0 = b*rho
                x1 = (midPointX + r*(-b))
                y1 = (midPointY + r*(a))
                x2 = (midPointX - r*(-b))
                y2 = (midPointY - r*(a))
                x1Sum = x1Sum + x1
                x2Sum = x2Sum + x2
                y1Sum = y1Sum + y1
                y2Sum = y2Sum + y2
                numLines = numLines + 1
    if numLines != 0:
        cv2.line(image, (int(x1Sum/numLines), int(y1Sum/numLines)), (int(x2Sum/numLines), int(y2Sum/numLines)), (255, 0, 0), 5)
    return image    
        

# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0

    # background subtractor
    backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

    # cleaning image kernel
    kernel = np.ones((3,3), np.uint8)
	
    # checks whether frames were extracted 
    success, image = vidObj.read() 

    if image is None:
        raise RuntimeError("Image not found")

    # Video writer
    height, width, layers = image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("output" + path[7] + ".mp4", fourcc, 30, (width, height))

    # Processing the image
    while success: 
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read()

        # foreground
        foreground = backgroundSubtractor.apply(image)
        if foreground is None:
            break

        # Processing the image of clean it
        cleanedImage = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

        # Canny image detector
        # Modify lower and upper threshold for better edges
        cannyImage = cv2.Canny(cleanedImage, 20, 120, apertureSize = 3)
        
        # Getting line length and mid point
        (midPointX, midPointY, lineLength) = getMidPointAndLength(cannyImage)
        
        # Drawing lines on the image
        image = drawLines(cannyImage, image, midPointX, midPointY, lineLength)
        
        # Limiting the number of frames 	        # Limiting the number of frames
        # if count == 300:
        #     break

        # Displaying the part in focus
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600, 600)
        cv2.imshow('image', image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        # video.write(image)

    video.release()
    vidObj.release()
    cv2.destroyAllWindows()
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("Videos/1.mp4")