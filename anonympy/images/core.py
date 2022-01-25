import os
import cv2
import random
import numpy as np

from utils import pixelated
from utils import sap_noise
from utils import find_middle, find_radius


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
          self.scaleFactor = 1.1
          self.minNeighbors = 5


     def resize(self, new_width=500):
         height, width, _ = self.frame.shape
         ratio = height / width
         new_height = int(ratio * new_width)
         return cv2.resize(self.frame, (new_width, new_height))


     def face_blur(self, kernel = (15,15), shape = 'c', box = None):
          '''
          Apply Gaussian Blur to the Face
          
          Parameters:
          ----------

          Returns:
          ----------

          Raises:
          ----------

          Examples
          ---------- 
               '''
          self.detections = self.FACE.detectMultiScale(self.frame,scaleFactor = self.scaleFactor, minNeighbors = self.minNeighbors)
          
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

     
     def face_SaP(self, kernel = (15,15), shape = 'c', box = None):
          '''
          Add Salt and Pepper Noise
          
          Parameters:
          ----------

          Returns:
          ----------

          Raises:
          ----------

          Examples
          ---------- 
          '''
          self.detections = self.FACE.detectMultiScale(self.frame,scaleFactor = self.scaleFactor, minNeighbors = slef.minNeighbors)
          
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


     def face_pixel(self, blocks = 20, shape = 'c', box = None):
          '''
          Add Pixelated Bluring to Face

          Parameters:
          ----------

          Returns:
          ----------

          Raises:
          ----------

          Examples
          ---------- 
          '''
          self.detections = self.FACE.detectMultiScale(self.frame,scaleFactor = self.scaleFactor, minNeighbors = self.minNeighbors)
          
          for face in self.detections:
               x,y,w,h = face

               noise = pixelated(self.frame[y:y+h,x:x+w], blocks = blocks)

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


     def blur(self, method='Gaussian', kernel=(15, 15)):
          '''
          Apply blurring to image. Available methods:
          - Averaging
          - Gaussian 
          - Bilateral 
          - Median 

          Parameters:
          ----------

          Returns:
          ----------

          Raises:
          ----------

          Examples
          ---------- 
          
          '''
          if method.lower() == 'gaussian':
               return cv2.GaussianBlur(self.frame, kernel, cv2.BORDER_DEFAULT) 
          elif method.lower() == 'median':
               if type(kernel) == tuple:
                    ksize = kernel[0]
               else:
                    ksize = kernel
               return  cv2.medianBlur(self.frame, ksize)
          elif method.lower() == 'bilateral':
               return  cv2.bilateralFilter(self.frame, *kernel)
          elif method.lower() == 'averaging':
               return cv2.blur(self.frame, kernel)


     def imshow(self, fname: str):
          '''
          Display the Image 
          '''
          cv2.imshow(fname, self.frame)


##img[y:y+h,x:x+w] = cv2.cvtColor(add_noise(cv2.cvtColor(img[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR )

## maskkkk
##x,y,w,h = 389, 127,209,209
##
##noised = add_noise(img[y:y+h,x:x+w])
##new = img.copy()
##new[y:y+h,x:x+w]  = noised
##
##mask = np.zeros(new.shape[:2], dtype='uint8')
##cv2.circle(mask, (493, 231), 105, 255, -1)
##
##img[mask > 0] = new[mask > 0]
##cv2.imshow('a', img)





anonym = imAnonymizer('me.png')

gaussian = anonym.blur(method = 'gaussian', kernel = (21, 21))
avg = anonym.blur(method = 'averaging', kernel = (15, 15))
median = anonym.blur(method = 'median', kernel = 11)
bilateral = anonym.blur(method = 'bilateral', kernel = (30, 150, 150))


cv2.imshow('gaussian', gaussian)
cv2.imshow('averaging', avg)
cv2.imshow('median', median)
cv2.imshow('bilateral', bilateral)































