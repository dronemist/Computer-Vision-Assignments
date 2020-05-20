# Medial axis detection of moving objects
In this assignment we used the openCV operations to highlight the medial axis of a moving object in a video.
[more](problem_statement.pdf)
## Opertions performed
1) _Background Subtraction_
    - In this part we used openCV in order to seperate the moving/foreground object from the background.
    - We also removed the noise from the obtained image to produce better results.
2) _Edges and lines_
    - We used the derivatives to identify the edges and then used the [hough line transform](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html), to detect lines.
3) _Medial axis detection_
    - To get the medial line, we took the average of all the detected lines.
    - In order to get the length of the moving object, we took the egde detected with minimum and maximum y-coordinate and used them to obtain the length.

## Results
This was the result obtained on one of the videos:
<p>
  <img src="https://github.com/dronemist/Computer-Vision-Assignments/blob/master/Assignment%201/Images/final/result_1.png">
</p>
