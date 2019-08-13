# Program to extract frames
import cv2
import numpy as np
# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
    
    # background subtractor
    backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        
        # foreground
        foreground = backgroundSubtractor.apply(image)

        # Processing the image of clean it
        kernel = np.ones((5,5),np.uint8)
        cleanedImage = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

        # Canny image detector
        # Modify lower and upper threshold for better edges
        cannyImage = cv2.Canny(cleanedImage, 50, 100, apertureSize = 3)
        
        # Hough lines 
        lines = cv2.HoughLines(cannyImage, 1, np.pi/180, 200)

        if lines is not None:
            # Applying hough transform
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Saves the frames with frame-count 
        cv2.imwrite("Images/final/frame%d.jpg" % count, image) 
        count += 1

        # Limiting the number of frames 
        if count == 30:
            break
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("Videos/1.mp4")