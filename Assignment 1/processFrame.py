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

    # background subtractor MOG
    bs2 = cv2.bgsegm.createBackgroundSubtractorMOG()
    # cleaning image kernel

    kernel = np.ones((3,3), np.uint8)

    # making the background subtractor learn
    # vidObjLearn = cv2.VideoCapture(path)
    # countLearn = 0
    # successLearn, imageLearn = vidObjLearn.read()
    # while successLearn:
    #
    #     countLearn += 1
    #
    #     successLearn, imageLearn = vidObjLearn.read()
    #     foregroundLearn = backgroundSubtractor.apply(imageLearn, learningRate = 0.01)
	
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
        # ddepth = cv2.CV_16S
        # # Canny image detector
        # # Modify lower and upper threshold for better edges
        # grad_x = cv2.Sobel(cleanedImage, ddepth, 1, 0, ksize=3)
        # grad_y = cv2.Sobel(cleanedImage, ddepth, 0, 1, ksize=3)
        # abs_grad_x = cv2.convertScaleAbs(grad_x)
        # abs_grad_y = cv2.convertScaleAbs(grad_y)
        # grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        cannyImage = cv2.Canny(cleanedImage, 10, 90, apertureSize = 3)
        
        # Hough lines 
        # lines = cv2.HoughLines(cannyImage, 1, np.pi/180, 120)
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(cannyImage, 1, np.pi / 180, 20, minLineLength, maxLineGap)
        # temp = 0

        # r = EndPoints(cannyImage, lines[0])
        
        # Taking the average of the lines
        # if np.shape(lines) == ():
        #     numLines = 0
        # else:
        #     numLines = (np.shape(lines))[0]
        # x1Sum = 0
        # y1Sum = 0
        # x2Sum = 0
        # y2Sum = 0
        # if lines is not None:
        #     # Applying hough transform
        #     r = 875
        #     for line in lines:
        #         for rho, theta in line:
        #             a = np.cos(theta)
        #             b = np.sin(theta)
        #             x0 = a*rho
        #             y0 = b*rho
        #             x1 = int(x0 + r*(-b))
        #             y1 = int(y0 + r*(a))
        #             x2 = int(x0 - r*(-b))
        #             y2 = int(y0 - r*(a))
        #             temp = temp + 1
        #             x1Sum = x1Sum + x1
        #             x2Sum = x2Sum + x2
        #             y1Sum = y1Sum + y1
        #             y2Sum = y2Sum + y2
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
                    x1Min = min(x1Min, x1)
                    y1Min = min(y1Min, y1)
                    x2Max = max(x2Max, x2)
                    y2Max = max(y2Max, y2)
                    if x2-x1 == 0:
                        angleSum = angleSum + np.arctan(np.inf)
                    else:
                        angleSum = angleSum + np.arctan((y2-y1)/(x2-x1))
                    numLines = numLines + 1

                    # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)


        except:
            # do nothing
            temp += 1
        # print (lines)
        # print(np.shape(lines))
        # print(numLines)
        midPointX = ((x1Min + x2Max) / 2)
        midPointY = ((y1Min + y2Max) / 2)
        if numLines != 0:
            # print(numLines)
            angleAvg = angleSum/numLines
            if x1-midPointX == 0:
                anglePoint1 = np.pi/2
            else:
                anglePoint1 = np.arctan((y1-midPointY)/(x1-midPointX))
            if x2-midPointX == 0:
                anglePoint2 = np.pi/2
            else:
                anglePoint2 = np.arctan((y2 - midPointY) / (x2 - midPointX))
            relativePoint1X = x1 - midPointX
            relativePoint1Y = y1 - midPointY
            relativePoint2X = x2 - midPointX
            relativePoint2Y = y2 - midPointY
            relativeAnglePoint1 = anglePoint1 - angleAvg
            relativeAnglePoint2 = anglePoint2 - angleAvg
            projectedX1 =  int(midPointX + (relativePoint1X)*(np.cos(relativeAnglePoint1))*(np.cos(angleAvg))/(np.cos(anglePoint1)))
            projectedX2 = int(midPointX + (relativePoint2X) * (np.cos(relativeAnglePoint2)) * (np.cos(angleAvg)) / (
                np.cos(anglePoint2)))
            projectedY1 = int(midPointY + (relativePoint1X) * (np.cos(relativeAnglePoint1)) * (np.sin(angleAvg)) / (
                np.cos(anglePoint1)))
            projectedY2 = int(midPointY + (relativePoint2X) * (np.cos(relativeAnglePoint2)) * (np.sin(angleAvg)) / (
                np.cos(anglePoint2)))

            # cv2.line(image, (projectedX1, projectedY1), (projectedX2, projectedY2), (255, 0, 0), 2)
            cv2.line(image, (int(x1Min), int(y1Min)), (int(x2Max), int(y2Max)), (255, 0, 0), 2)
            # print((int(x1Sum/numLines), y1Min), (int(x2Sum/numLines), y2Max))
            # cv2.line(image, (int(x1Sum/numLines), y1Min), (int(x2Sum/numLines), y2Max), (255, 0, 0), 2)
        # # Converting it back to video
        # # Uncomment this line to write to video
        video.write(image)
        # cv2.imwrite("Images/final/frame%d.jpg" % count, image)
        # count += 1

        # print(temp)
        # Limiting the number of frames 	        # Limiting the number of frames
        # if count == 50:
        #     break

    # print(temp)
    video.release()
    vidObj.release()
    # vidObjLearn.release()
    cv2.destroyAllWindows()
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("Videos/1.mp4")