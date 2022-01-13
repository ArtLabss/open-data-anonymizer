import cv2
import numpy as np
import os
import random
from utils import find_middle, find_radius, sap_noise 

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
     def __init__(self, path, flag = None):
          self.path = path
          self.flag = flag
          self.frame = cv2.imread(path, flag)
          self.FACE = cv2.CascadeClassifier(r"utils/cascade.xml")


     def resize(self, new_width=500):
         height, width, _ = self.frame.shape
         ratio = height / width
         new_height = int(ratio * new_width)
         return cv2.resize(self.frame, (new_width, new_height))


     def blur_face(self, kernel = (15,15), scaleFactor = 1.1, minNeighbors =5, shape = 'c', box = None):
          '''
          Apply Gaussian Blur to the Face 
          '''
          self.detections = self.FACE.detectMultiScale(self.frame,scaleFactor = scaleFactor, minNeighbors = minNeighbors)
          
          for face in self.detections:
               x,y,w,h = face
               
               noise = cv2.GaussianBlur(self.frame[y:y+h,x:x+w], kernel, cv2.BORDER_DEFAULT)

               if shape == 'c':
                    # circular
                    new = self.frame.copy()
                    new[y:y+h,x:x+w] = noise

                    #mask
                    mask = np.zeros(new.shape[:2], dtype='uint8')
                    # cirlce parameters 
                    cv2.circle(mask, find_middle(x,y,w,h), find_radius(x,y,w,h), 255, -1)

                    #apply
                    self.frame[mask > 0] = new[mask > 0]
                    
               elif shape == 'r':
                    # rectangular
                    self.frame[y:y+h,x:x+w] = noise
               
               if box == 'r':
                    cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)
               elif box == 'c':
                    cv2.circle(self.frame, find_middle(x,y,w,h), find_radius(x,y,w,h), (255,0,0), 2)

     
     def SaP_face(self, kernel = (15,15), scaleFactor = 1.1, minNeighbors =5, shape = 'c', box = None):
          '''
          Add Salt and Pepper Noise
          '''
          self.detections = self.FACE.detectMultiScale(self.frame,scaleFactor = scaleFactor, minNeighbors = minNeighbors)
          
          for face in self.detections:
               x,y,w,h = face

               noise = sap_noise(self.frame[y:y+h,x:x+w])

               if shape == 'c':
                    # circular
                    new = self.frame.copy()
                    new[y:y+h,x:x+w] = noise

                    #mask
                    mask = np.zeros(new.shape[:2], dtype='uint8')
                    # cirlce parameters 
                    cv2.circle(mask, find_middle(x,y,w,h), find_radius(x,y,w,h), 255, -1)

                    #apply
                    self.frame[mask > 0] = new[mask > 0]
                    
               elif shape == 'r':
                    # rectangular
                    self.frame[y:y+h,x:x+w] = noise
               
               if box == 'r':
                    cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)
               elif box == 'c':
                    cv2.circle(self.frame, find_middle(x,y,w,h), find_radius(x,y,w,h), (255,0,0), 2)
          

     
     def imshow(self, fname: str):
          '''
          Display the Image 
          '''
          cv2.imshow(fname, self.frame)



