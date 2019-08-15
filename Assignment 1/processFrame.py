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
    success, image = vidObj.read() 

    # Video writer
    height, width, layers = image.shape
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter('output2.mp4', fourcc, 15, (width, height))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('output2.avi', fourcc, 15, (width, height))
    # background subtractor
    backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()

    # cleaning image kernel
    kernel = np.ones((3,3), np.uint8)
    vidObj_temp = cv2.VideoCapture(path)
    count_temp = 0	
    success_temp, image_temp = vidObj.read()
    while success_temp:
        success_temp, image = vidObj_temp.read()
        foreground = backgroundSubtractor.apply(image, learningRate = 0.01)
        if count_temp == 50:
            break
	
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        
        # foreground
        foreground = backgroundSubtractor.apply(image, learningRate = 0.01)

        # Processing the image of clean it
        cleanedImage = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

        # Canny image detector
        # Modify lower and upper threshold for better edges
        cannyImage = cv2.Canny(cleanedImage, 10, 90, apertureSize = 3)
        
        # Hough lines 
        lines = cv2.HoughLines(cannyImage, 1, np.pi/180, 150)
        temp = 0
        # print(np.shape(lines))
        if np.shape(lines) == ():
            num_lines = 0
        else:
            num_lines = (np.shape(lines))[0]
        x1_sum = 0
        y1_sum = 0
        x2_sum = 0
        y2_sum = 0		
        if lines is not None:
            # Applying hough transform
            r = 875
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + r*(-b))
                    y1 = int(y0 + r*(a))
                    x2 = int(x0 - r*(-b))
                    y2 = int(y0 - r*(a))
                    temp = temp + 1
                    x1_sum = x1_sum + x1
                    x2_sum = x2_sum + x2
                    y1_sum = y1_sum + y1
                    y2_sum = y2_sum + y2
        if num_lines != 0:		
            cv2.line(image, (int(x1_sum/num_lines), int(y1_sum/num_lines)), (int(x2_sum/num_lines), int(y2_sum/num_lines)), (255, 0, 0), 2)

        # Converting it back to video
        # Uncomment this line to write to video
        # video.write(image)

        # Saves the frames with frame-count 
        cv2.imwrite("Images/final/frame%d.jpg" % count, image) 
        count += 1
        # print(temp)
        # Limiting the number of frames 
        if count == 50:
            break
    video.release()
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("Videos/1.mp4")