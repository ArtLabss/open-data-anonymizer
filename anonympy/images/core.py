import cv2
import numpy as np
import os
import random


class imAnonymizer(object):
     """
     Initialize an image as a imAnonymizer object

     Parameters:
     ----------

     Returns:
     ----------

     Raises:
     ----------

     Examples
     ---------- 
     """
     def __init__(self, frame):
          self.frame = frame.copy()


     def resize(self, new_width=500):
         height, width, _ = self.frame.shape
         ratio = height / width
         new_height = int(ratio * new_width)
         return cv2.resize(self.frame, (new_width, new_height))


     def blur_face(self, kernel = (15,15), scaleFactor = 1.1, minNeighbors =5, box = False):
          '''
          Apply Gaussian Blur to the Face 
          '''
          face_cascade = cv2.CascadeClassifier(r"utils/cascade.xml")
          detections = face_cascade.detectMultiScale(self.frame,scaleFactor = scaleFactor, minNeighbors = minNeighbors)
          print(detections)
          for face in detections:
               x,y,w,h = face

               self.frame[y:y+h,x:x+w] = cv2.GaussianBlur(self.frame[y:y+h,x:x+w], kernel, cv2.BORDER_DEFAULT)

               if box:
                    cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)


     def imshow(self, fname: str):
          '''
          '''

          cv2.imshow(fname, self.frame)
          
def add_noise(frame):
    img = frame.copy()
    # Getting the dimensions of the image
    row , col, _ = img.shape
     
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to white
        img[y_coord][x_coord] = 255
         
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to black
        img[y_coord][x_coord] = 0
         
    return img

##img[y:y+h,x:x+w] = cv2.cvtColor(add_noise(cv2.cvtColor(img[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR )

##def radius(x,y, w, h) -> tuple:
##	pt1 = (x, y)
##	pt2 = (x+w, y+h)
##
##	side_middle = x + w, (y + y + h) / 2
##	center = find_middle(pt1, pt2)
##	dis = side_middle[0] - center[0]
##	
##	return dis
     

##def find_middle(pt1: tuple, pt2: tuple) -> tuple:
##	x1, y1 = pt1
##	x2, y2 = pt2
##	m1, m2 = int((x1 + x2)/2), int((y1 + y2)/2)
##	return m1, m2
