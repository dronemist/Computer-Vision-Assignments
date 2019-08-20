# Program to extract frames
import cv2
import numpy as np

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

    # Video writer
    # foreground = backgroundSubtractor.apply(image)
    height, width, layers = image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('output.mp4', fourcc, 30, (width, height))
    temp = 0

    # Processing the image
    while success: 
        numLines = 0
        angleSum = 0
        midPointX = 0
        midPointY = 0
        x1Min = 1e6
        x2Max = -1e6
        y1Min = 1e6
        y2Max = -1e6
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read()
        # foreground
        foreground = backgroundSubtractor.apply(image)
        if foreground is None:
            break

        # Processing the image of clean it
        cleanedImage = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        # ddepth = cv2.CV_16S
        # # Canny image detector
        # # Modify lower and upper threshold for better edges
        cannyImage = cv2.Canny(cleanedImage, 20, 120, apertureSize = 3)
        
        # Hough lines 
        # lines = cv2.HoughLines(cannyImage, 1, np.pi/180, 120)
        minLineLength = 25
        maxLineGap = 10
        lines = cv2.HoughLinesP(cannyImage, 1, np.pi / 180, 100, minLineLength, maxLineGap)

        # temp = 0

        x1Sum = 0
        y1Sum = 0
        x2Sum = 0
        y2Sum = 0
        # if numLines != 0:
        #     cv2.line(image, (int(x1Sum/numLines), int(y1Sum/numLines)), (int(x2Sum/numLines), int(y2Sum/numLines)), (255, 0, 0), 2)
        try:
            if (lines[0] is not None):
                # numLines = np.shape(lines[0])
                for line in lines:
                    # print(line)
                    x1 = line[0,0]
                    y1 = line[0,1]
                    x2 = line[0,2]
                    y2 = line[0,3]
                    # x1 = 0
                    # x2 = 0
                    # y1 = 0
                    # y2 = 0
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
                    # numLines = numLines + 1
                    x1Sum = x1Sum + x1
                    y1Sum = y1Sum + y1
                    x2Sum = x2Sum + x2
                    y2Sum = y2Sum + y2
                    # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    numLines = numLines + 1
                    # cv2.line(image, (x1, y1),
                    #          (x2, y2), (0, 0, 255), 5)


        except:
            # do nothing
            temp += 1

        # print (lines)
        # print(np.shape(lines))
        # print(numLines)
        if numLines != 0:
            midPointX = (x1Sum + x2Sum) / (2*numLines)
            midPointY = (y1Sum + y2Sum) / (2*numLines)
        lineLength = np.sqrt((x1Min-x2Max)*(x1Min-x2Max) + (y1Min-y2Max)*(y1Min-y2Max))

        # Taking the average of the lines
        x1Sum = 0
        y1Sum = 0
        x2Sum = 0
        y2Sum = 0
        numLines = 0
        normalHoughLines = cv2.HoughLines(cannyImage, 1, np.pi / 180, 150)
        if normalHoughLines is not None:
            # Applying hough transform
            r = lineLength/2
            print(lineLength)
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
                    temp = temp + 1
                    x1Sum = x1Sum + x1
                    x2Sum = x2Sum + x2
                    y1Sum = y1Sum + y1
                    y2Sum = y2Sum + y2
                    numLines = numLines + 1
        if numLines != 0:
            cv2.line(image, (int(x1Sum/numLines), int(y1Sum/numLines)), (int(x2Sum/numLines), int(y2Sum/numLines)), (255, 0, 0), 5)
        # if numLines != 0:
        #     # print(numLines)
        #     angleAvg = angleSum/numLines
        #     if x1Min-midPointX == 0:
        #         anglePoint1 = -np.pi/2
        #     else:
        #         anglePoint1 = -np.arctan((y1Min-midPointY)/(x1Min-midPointX))
        #     if x2Max-midPointX == 0:
        #         anglePoint2 = -np.pi/2
        #     else:
        #         anglePoint2 = -np.arctan((y2Max - midPointY) / (x2Max - midPointX))
        #     relativePoint1X = x1Min - midPointX
        #     relativePoint1Y = y1Min - midPointY
        #     relativePoint2X = x2Max - midPointX
        #     relativePoint2Y = y2Max - midPointY
        #     relativeAnglePoint1 = anglePoint1 - angleAvg
        #     relativeAnglePoint2 = anglePoint2 - angleAvg
        #     if (np.cos(anglePoint1)!=0) and (np.cos(anglePoint2)!=0):
        #         projectedX1 =  int(midPointX + (relativePoint1X)*(np.cos(relativeAnglePoint1))*(np.cos(angleAvg))/(np.cos(anglePoint1)))
        #         projectedX2 = int(midPointX + (relativePoint2X) * (np.cos(relativeAnglePoint2)) * (np.cos(angleAvg)) / (
        #             np.cos(anglePoint2)))
        #         projectedY1 = int(midPointY - (relativePoint1X) * (np.cos(relativeAnglePoint1)) * (np.sin(angleAvg)) / (
        #             np.cos(anglePoint1)))
        #         projectedY2 = int(midPointY - (relativePoint2X) * (np.cos(relativeAnglePoint2)) * (np.sin(angleAvg)) / (
        #             np.cos(anglePoint2)))
        #         try:
        #             cv2.line(image, (projectedX1, projectedY1), (projectedX2, projectedY2), (255, 0, 0), 2)
        #         except:
        #             print((projectedX1, projectedY1), (projectedX2, projectedY2))
            # cv2.line(image, (int(x1Min), int(y1Min)), (int(x2Max), int(y2Max)), (255, 0, 0), 2)
            # print((int(x1Sum/numLines), y1Min), (int(x2Sum/numLines), y2Max))
            # cv2.line(image, (int(x1Sum/numLines), y1Min), (int(x2Sum/numLines), y2Max), (255, 0, 0), 2)
        # # Converting it back to video
        # # Uncomment this line to write to video
        # video.write(image)
        # cv2.imwrite("Images/final/frame%d.jpg" % count, image)
        # count += 1

        # print(temp)
        # Limiting the number of frames 	        # Limiting the number of frames
        # if count == 300:
        #     break

        # Displaying the part in focus
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600, 600)
        cv2.imshow('image', image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # print(temp)
    video.release()
    vidObj.release()
    # vidObjLearn.release()
    cv2.destroyAllWindows()
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("Videos/2.mp4")