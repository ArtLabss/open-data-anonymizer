import os
import cv2
import glob
import shutil
import random
import numpy as np

from utils import pixelated
from utils import sap_noise
from utils import find_middle, find_radius


class imAnonymizer(object):
     """
     Initialize an image/directory as a imAnonymizer object

     Parameters:
     ----------

     Returns:
     ----------

     Raises:
     ----------

     Examples
     ---------- 
     """
     def __init__(self, path, dst = None):
          if os.path.isdir(path):
               self.path = path     
               self._path = True
               self._img = False
          else:
               self.frame = path.copy()
               self._path = False
               self._img = True

          if dst is not None:
               self.dst = dst
               self._dst = True
          else:
               self._dst = False
                    
          self.FACE = cv2.CascadeClassifier("utils\cascade.xml")
          self.scaleFactor = 1.1
          self.minNeighbors = 5


     def resize(self, new_width=500):
         height, width, _ = self.frame.shape
         ratio = height / width
         new_height = int(ratio * new_width)
         return cv2.resize(self.frame, (new_width, new_height))


     def _face_blur(self, img, kernel = (15,15), shape = 'c', box = None):
          '''
          '''
          self.detections = self.FACE.detectMultiScale(img,scaleFactor = self.scaleFactor, minNeighbors = self.minNeighbors)
     
          for face in self.detections:
               x,y,w,h = face
               
               noise = cv2.GaussianBlur(img[y:y+h,x:x+w], kernel, cv2.BORDER_DEFAULT)
               copy = img.copy()
               
               if shape == 'c':
                    # circular
                    new = img.copy()
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
                    
               else:
                    raise Exception('Possible values: `r` (rectangular) and `c` (circular)')
                    
               
               if box == 'r':
                    cv2.rectangle(copy, (x,y) ,(x+w,y+h), (255,0,0), 2)
               elif box == 'c':
                    cv2.circle(copy, find_middle(x,y,w,h), find_radius(x,y,w,h), (255,0,0), 2)
               elif box is None:
                    pass
               else:
                    raise Exception('Possible values: `r` (rectangular) and `c` (circular), default `None`')
                    
          return copy


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
          if self._img:
               return self._face_blur(self.frame, kernel = kernel, shape = shape, box = box)
                    
          elif self._path:
               for filepath in glob.iglob(self.path + "/**/*.*", recursive=True):
                    # Ignore non images
                    if not filepath.endswith((".png", ".jpg", ".jpeg")):
                         continue
                    # Process Image
                    img = cv2.imread(filepath)
                    img = self._face_blur(img, shape = shape, box = box)

                    output_filepath = filepath.replace(os.path.split(self.path)[1], 'Output')
                    output_dir = os.path.dirname(output_filepath)
                    # Ensure the folder exists
                    os.makedirs(output_dir, exist_ok=True)

                    cv2.imwrite(output_filepath, img)
               if self._dst:
                    data_from = self.path.replace(os.path.split(self.path)[1], 'Output')
                    data_to = os.path.join(self.dst, 'Output')
                    shutil.copytree(data_from, data_to, dirs_exist_ok = True)
                    shutil.rmtree(data_from)

     
##shutil.rmtree(out)
##shutil.copytree(out, dst, dirs_exist_ok = True)

     def _face_SaP(self, img, shape = 'c', box = None):
          '''
          '''
          self.detections = self.FACE.detectMultiScale(img, scaleFactor = self.scaleFactor, minNeighbors = self.minNeighbors)
          
          for face in self.detections:
               x,y,w,h = face

               noise = sap_noise(img[y:y+h,x:x+w])
               copy = img.copy()

               if shape == 'c':
                    # circular
                    new = img.copy()
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

               else:
                    raise Exception('Possible values: `r` (rectangular) and `c` (circular)')
               
               if box == 'r':
                    cv2.rectangle(copy, (x,y),(x+w,y+h),(255,0,0),2)
               elif box == 'c':
                    cv2.circle(copy, find_middle(x,y,w,h), find_radius(x,y,w,h), (255,0,0), 2)
               elif box == None:
                    pass
               else:
                    raise Exception('Possible values: `r` (rectangular) and `c` (circular), default `None`')
          return copy


     def face_SaP(self, shape = 'c', box = None):
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
          if self._img:
               return self._face_SaP(self.frame, shape = shape, box = box)
                    
          elif self._path:
               for filepath in glob.iglob(self.path + "/**/*.*", recursive=True):
                    # Ignore non images
                    if not filepath.endswith((".png", ".jpg", ".jpeg")):
                         continue
                    # Process Image
                    img = cv2.imread(filepath)
                    img = self._face_SaP(img, shape = shape, box = box)

                    output_filepath = filepath.replace(os.path.split(self.path)[1], 'Output')
                    output_dir = os.path.dirname(output_filepath)
                    # Ensure the folder exists
                    os.makedirs(output_dir, exist_ok=True)

                    cv2.imwrite(output_filepath, img)
               if self._dst:
                    data_from = self.path.replace(os.path.split(self.path)[1], 'Output')
                    data_to = os.path.join(self.dst, 'Output')
                    shutil.copytree(data_from, data_to, dirs_exist_ok = True)
                    shutil.rmtree(data_from)


     def _face_pixel(self, img, blocks = 20, shape = 'c', box = None):
          '''
          '''
          self.detections = self.FACE.detectMultiScale(img, scaleFactor = self.scaleFactor, minNeighbors = self.minNeighbors)
          
          for face in self.detections:
               x,y,w,h = face

               noise = pixelated(img[y:y+h,x:x+w], blocks = blocks)
               copy = img.copy()

               if shape == 'c':
                    # circular
                    new = img.copy()
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
                    
               else:
                    raise Exception('Possible values: `r` (rectangular) and `c` (circular)')
               
               if box == 'r':
                    cv2.rectangle(copy, (x,y), (x+w,y+h), (255,0,0), 2)
               elif box == 'c':
                    cv2.circle(copy, find_middle(x,y,w,h), find_radius(x,y,w,h), (255,0,0), 2)
               elif box is None:
                    pass
               else:
                    raise Exception('Possible values: `r` (rectangular) and `c` (circular), default `None`')

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
          if self._img:
               return self._face_pixel(self.frame, blocks = blocks, shape = shape, box = box)
                    
          elif self._path:
               for filepath in glob.iglob(self.path + "/**/*.*", recursive=True):
                    # Ignore non images
                    if not filepath.endswith((".png", ".jpg", ".jpeg")):
                         continue
                    # Process Image
                    img = cv2.imread(filepath)
                    img = self._face_pixel(img, blocks = blocks, shape = shape, box = box)

                    output_filepath = filepath.replace(os.path.split(self.path)[1], 'Output')
                    output_dir = os.path.dirname(output_filepath)
                    # Ensure the folder exists
                    os.makedirs(output_dir, exist_ok=True)

                    cv2.imwrite(output_filepath, img)

               if self._dst:
                    data_from = self.path.replace(os.path.split(self.path)[1], 'Output')
                    data_to = os.path.join(self.dst, 'Output')
                    shutil.copytree(data_from, data_to, dirs_exist_ok = True)
                    shutil.rmtree(data_from)
          

     def _blur(self, img, method='Gaussian', kernel=(15, 15)):
          '''          
          '''
          if method.lower() == 'gaussian':
               return cv2.GaussianBlur(img, kernel, cv2.BORDER_DEFAULT) 
          elif method.lower() == 'median':
               if type(kernel) == tuple:
                    ksize = kernel[0]
               else:
                    ksize = kernel
               return  cv2.medianBlur(img, ksize)
          elif method.lower() == 'bilateral':
               return  cv2.bilateralFilter(img, *kernel)
          elif method.lower() == 'averaging':
               return cv2.blur(img, kernel)


     def blur(self, method='Gaussian', kernel=(15, 15)):
          '''
          Apply blurring to image. Available methods:
          - Averaging (ex.kernel = (15, 15))
          - Gaussian (ex. kernel = (21, 21))
          - Bilateral  (ex. kernel = (30, 150, 150))
          - Median  (ex. kernel = 11)

          Parameters:
          ----------

          Returns:
          ----------

          Raises:
          ----------

          Examples
          ---------- 
          '''
          if self._img:
               return self._blur(self.frame, method = method, kernel= kernel)
                    
          elif self._path:
               for filepath in glob.iglob(self.path + "/**/*.*", recursive=True):
                    # Ignore non images
                    if not filepath.endswith((".png", ".jpg", ".jpeg")):
                         continue
                    # Process Image
                    img = cv2.imread(filepath)
                    img = self._blur(img, method= method, kernel= kernel)

                    output_filepath = filepath.replace(os.path.split(self.path)[1], 'Output')
                    output_dir = os.path.dirname(output_filepath)
                    # Ensure the folder exists
                    os.makedirs(output_dir, exist_ok=True)

                    cv2.imwrite(output_filepath, img)
               if self._dst:
                    data_from = self.path.replace(os.path.split(self.path)[1], 'Output')
                    data_to = os.path.join(self.dst, 'Output')
                    shutil.copytree(data_from, data_to, dirs_exist_ok = True)
                    shutil.rmtree(data_from)
