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
     def __init__(self, path, output = None):
          if os.path.isdir(path):
               self.path = path
               self._path = True
               self._img = False
          else:
               self.frame = path.copy()
               self._path = False
               self._img = True
                    
          self.FACE = cv2.CascadeClassifier("utils\cascade.xml")
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
               copy = self.frame.copy()
               if shape == 'c':
                    # circular
                    new = self.frame.copy()
                    new[y:y+h,x:x+w] = noise

                    #mask
                    mask = np.zeros(new.shape[:2], dtype='uint8')
                    # cirlce parameters 
                    cv2.circle(mask, find_middle(x,y,w,h), find_radius(x,y,w,h), 255, -1)

                    #apply
                    copy[mask > 0] = new[mask > 0]
                    
               elif shape == 'r':
                    # rectangular
                    copy[y:y+h,x:x+w] = noise
               
               if box == 'r':
                    cv2.rectangle(copy, (x,y) ,(x+w,y+h), (255,0,0), 2)
               elif box == 'c':
                    cv2.circle(copy, find_middle(x,y,w,h), find_radius(x,y,w,h), (255,0,0), 2)
          return copy     

     
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
          self.detections = self.FACE.detectMultiScale(self.frame,scaleFactor = self.scaleFactor, minNeighbors = self.minNeighbors)
          
          for face in self.detections:
               x,y,w,h = face

               noise = sap_noise(self.frame[y:y+h,x:x+w])
               copy = self.frame.copy()

               if shape == 'c':
                    # circular
                    new = self.frame.copy()
                    new[y:y+h,x:x+w] = noise

                    #mask
                    mask = np.zeros(new.shape[:2], dtype='uint8')
                    # cirlce parameters 
                    cv2.circle(mask, find_middle(x,y,w,h), find_radius(x,y,w,h), 255, -1)

                    #apply
                    copy[mask > 0] = new[mask > 0]
                    
               elif shape == 'r':
                    # rectangular
                    copy[y:y+h,x:x+w] = noise
               
               if box == 'r':
                    cv2.rectangle(copy, (x,y),(x+w,y+h),(255,0,0),2)
               elif box == 'c':
                    cv2.circle(copy, find_middle(x,y,w,h), find_radius(x,y,w,h), (255,0,0), 2)
               else:
                    raise Exception('Possible values: `r` (rectangular) and `c` (circular)')
          return copy


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
               copy = self.frame.copy()

               if shape == 'c':
                    # circular
                    new = self.frame.copy()
                    new[y:y+h,x:x+w] = noise

                    #mask
                    mask = np.zeros(new.shape[:2], dtype='uint8')
                    # cirlce parameters 
                    cv2.circle(mask, find_middle(x,y,w,h), find_radius(x,y,w,h), 255, -1)

                    #apply
                    copy[mask > 0] = new[mask > 0]
                    
               elif shape == 'r':
                    # rectangular
                    copy[y:y+h,x:x+w] = noise
               
               if box == 'r':
                    cv2.rectangle(copy, (x,y), (x+w,y+h), (255,0,0), 2)
               elif box == 'c':
                    cv2.circle(copy, find_middle(x,y,w,h), find_radius(x,y,w,h), (255,0,0), 2)

          return copy


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





## for dirpath, dirnames, filenames in os.walk(inputpath):
##	print(dirpath, '\t', dirnames, '\t', filenames)




##for dirpath, dirnames, filenames in os.walk(inputpath):
##    structure = os.path.join(outputpath, dirpath[len(inputpath):])
##    if not os.path.isdir(structure):
##        os.mkdir(structure)
##    else:
##        print("Folder does already exits!")

##def check_folders(path, lst):
##	for file in os.listdir(path):
##		dst = os.path.join(path, file)
##		if os.path.isdir(dst):
##			return hack_location(dst, lst)
##		else:
##			if os.path.splitext(dst)[1].strip('.') in ('jpg', 'jpeg', 'png'):
##				lst.append(dst)















