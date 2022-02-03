import os
import cv2
import glob
import shutil
import random
import numpy as np
from typing import Union

from anonympy.images.utils import pixelated
from anonympy.images.utils import sap_noise
from anonympy.images.utils import find_middle, find_radius


class imAnonymizer(object):
     """
     Initialize an image or a directory as a imAnonymizer object

     Parameters:
     ----------
     path : Union[numpy.ndarray, str]
          cv2.imread object or path string to a folder.
     dst : str, default None
          destination to save the output folder if a string was passed to `path`. If `dst = None` a new
          foulder will be created in the same directory, else in the directory specified. 

     Returns:
     ----------
     imAnonymizer object
     
     Examples
     ----------
     >>> from anonympy.images import imAnonymizer

    Contructing imAnonymizer object by passing an image:

    >>> img = cv2.imread('C://Users/shakhansho/Downloads/image.png')
    >>> anonym = imAnonymizer(img)
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
                    
          self._FACE = cv2.CascadeClassifier("utils\cascade.xml")
          self.scaleFactor = 1.1
          self.minNeighbors = 5


     def _face_blur(self, img, kernel = (15,15), shape = 'c', box = None, fname = None):
          '''
          Function to apply Gaussian blur to an image
          '''
          self.detections = self._FACE.detectMultiScale(img,scaleFactor = self.scaleFactor, minNeighbors = self.minNeighbors)
          if self.detections == tuple():
               if self._img:
                    print(f'No Faces were Detected in the Image')
               elif self.path:
                    print(f'No Faces were Detected in the {fname}')
               return None
          else:
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
          Based on cv2.GaussianBlur.
          
          Parameters:
          ----------
          kernel : tuple, default (15, 15)
               Gaussian Kernel Size. [height width]. height and width should be odd and can have different values
          shape : str, default 'c'
               Blurring shape. Possible values: `r` (rectangular blurring) and `c` (circular blurring).
          box : str, default None 
               Bounding box. Possible values: `r` (rectangular) and `c` (circular), default `None`.

          Returns:
          ----------
          If an image was passed, a blurred one will be returned. If a path to a folder was passed, a similar folder
          with blurred images will be created. 
          
          Raises:
          ----------
          Exception:
               * if argument for `shape` is something other than `r` or `c` 
               * if argument for `box` is something other that `r`, `c` or None.

          Notes
          ----------
          For the method to work properly, a face has to be present in the image.
          If a path to a folder was passed to `imAnonymizer`, the method will create a new folder `Output`
          with blurred images, preserving the folder structure and image names.
          
          Examples
          ----------
          >>> from anonympy.images import imAnonymizer
          
          Blurring an image

          >>> img = cv2.imread('C://Users/shakhansho/Downloads/image.jpg')
          >>> anonym = imAnonymizer(img)
          >>> blurred = anonym.face_blur()

          Applying Gaussian Blur to all images in a folder

          >>> path = 'C://Users/shakhansho/Downloads/Images'
          >>> anonym = imAnonymizer(path, dst = 'D;//Output')
          >>> anonym.face_blur(shape = 'r', box = 'r')
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
                    img = self._face_blur(img, shape = shape, box = box, fname = filepath)

                    output_filepath = filepath.replace(os.path.split(self.path)[1], 'Output')
                    output_dir = os.path.dirname(output_filepath)
                    # Ensure the folder exists
                    os.makedirs(output_dir, exist_ok=True)

                    if img is None:
                         pass
                    else:
                         cv2.imwrite(output_filepath, img)
               if self._dst:
                    data_from = self.path.replace(os.path.split(self.path)[1], 'Output')
                    data_to = os.path.join(self.dst, 'Output')
                    shutil.copytree(data_from, data_to, dirs_exist_ok = True)
                    shutil.rmtree(data_from)


     def _face_SaP(self, img, shape = 'c', box = None, fname = None): 
          '''
          Function to apply Salt and Pepper Noise to an Image 
          '''
          self.detections = self._FACE.detectMultiScale(img, scaleFactor = self.scaleFactor, minNeighbors = self.minNeighbors)
          if self.detections == tuple():
               if self._img:
                    print(f'No Faces were Detected in the Image')
               elif self.path:
                    print(f'No Faces were Detected in the {fname}')
               return None
          else:
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
          Add Salt and Pepper Noise.
          
          Parameters:
          ----------
          shape : str, default 'c'
               Blurring shape. Possible values: `r` (rectangular blurring) and `c` (circular blurring).
          box : str, default None
               Bounding box. Possible values: `r` (rectangular) and `c` (circular), default `None`.
               
          Returns:
          ----------
          If an image was passed, an image wiht noise will be returned. If a path to a folder was passed, a similar folder
          with noised images will be created. 

          Raises:
          ----------
          Exception:
               * if argument for `shape` is something other than `r` or `c` 
               * if argument for `box` is something other that `r`, `c` or None.

          Notes
          ----------
          For the method to work properly, a face has to be present in the image.
          If a path to a folder was passed to `imAnonymizer`, the method will create a new folder `Output`
          with noise added to the images, preserving the folder structure and image names.

          Examples
          ----------
          >>> from anonympy.images import imAnonymizer
          
          Add noise to an image 

          >>> img = cv2.imread('C://Users/shakhansho/Downloads/image.jpg')
          >>> anonym = imAnonymizer(img)
          >>> noised = anonym.face_SaP(shape = 'c')

          Add noise to all images in a folder 

          >>> path = 'C://Users/shakhansho/Downloads/Images'
          >>> anonym = imAnonymizer(path, dst = 'D;//Output')
          >>> anonym.face_SaP(shape = 'c', box = None)
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
                    img = self._face_SaP(img, shape = shape, box = box, fname = filepath)

                    output_filepath = filepath.replace(os.path.split(self.path)[1], 'Output')
                    output_dir = os.path.dirname(output_filepath)
                    # Ensure the folder exists
                    os.makedirs(output_dir, exist_ok=True)

                    if img is None:
                         pass
                    else:
                         cv2.imwrite(output_filepath, img)
               if self._dst:
                    data_from = self.path.replace(os.path.split(self.path)[1], 'Output')
                    data_to = os.path.join(self.dst, 'Output')
                    shutil.copytree(data_from, data_to, dirs_exist_ok = True)
                    shutil.rmtree(data_from)


     def _face_pixel(self, img, blocks = 20, shape = 'c', box = None, fname = None):
          '''
          '''
          self.detections = self._FACE.detectMultiScale(img, scaleFactor = self.scaleFactor, minNeighbors = self.minNeighbors)
          if self.detections == tuple():
               if self._img:
                    print(f'No Faces were Detected in the Image')
               elif self.path:
                    print(f'No Faces were Detected in the {fname}')
               return None
          else:
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
          Add Pixelated Bluring to a Face

          Parameters:
          ----------
          blocks : int, default 20
               face image dimensions are divided into MxN blocks.
          shape : str, default 'c'
               Blurring shape. Possible values: `r` (rectangular blurring) and `c` (circular blurring).
          box : str, default None 
               Bounding box. Possible values: `r` (rectangular) and `c` (circular), default `None`.

          Returns:
          ----------
          If an image was passed, an image wiht pixaled blurred face will be returned. If a path to a folder was passed, a similar folder
          with noised images will be created. 

          Raises:
          ----------
          Exception:
               * if argument for `shape` is something other than `r` or `c` 
               * if argument for `box` is something other that `r`, `c` or None.

          Notes
          ----------
          For the method to work properly, a face has to be present in the image.
          If a path to a folder was passed to `imAnonymizer`, the method will create a new folder `Output`
          with blurred pixaled face, while preserving the folder structure and image names.
          
          Examples
          ----------
          >>> from anonympy.images import imAnonymizer
          
          Blur Face in an image 

          >>> img = cv2.imread('C://Users/shakhansho/Downloads/image.jpg')
          >>> anonym = imAnonymizer(img)
          >>> noised = anonym.face_pixel(shape = 'c')

          Blur all images in a folder 

          >>> path = 'C://Users/shakhansho/Downloads/Images'
          >>> anonym = imAnonymizer(path, dst = 'D;//Output')
          >>> anonym.face_pixel(shape = 'r', box = 'r')
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
                    img = self._face_pixel(img, blocks = blocks, shape = shape, box = box, fname = filepath)

                    output_filepath = filepath.replace(os.path.split(self.path)[1], 'Output')
                    output_dir = os.path.dirname(output_filepath)
                    # Ensure the folder exists
                    os.makedirs(output_dir, exist_ok=True)

                    if img is None:
                         pass
                    else:
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
          Apply blurring to image
          Based on OpenCV functions.

          Parameters:
          ----------
          method : str, default 'Gaussian'
               Available methods:
                    - Averaging.
                         Example kernel size (30, 30)
                    - Gaussian
                         Kernel (height, width). Height and Width should be odd and can have different values, example (15, 15).                          
                    - Bilateral
                         Example kernel size (30, 150, 150),
                    - Median
                         Example kernel size 11,
          kernel : tuple, default (15, 15)

          Returns:
          ----------
          If an image was passed, a blurred image will be returned. If a path to a folder was passed, a similar folder
          with blurred images will be created. 

          Examples
          ----------
          >>> from anonympy.images import imAnonymizer
          
          Blur an image 

          >>> img = cv2.imread('C://Users/shakhansho/Downloads/image.jpg')
          >>> anonym = imAnonymizer(img)
          >>> blurred = anonym.blur(method = 'gaussian', kernel = (21, 21))

          Blur all images in a folder 

          >>> path = 'C://Users/shakhansho/Downloads/Images'
          >>> anonym = imAnonymizer(path, dst = 'D;//Output')
          >>> anonym.blur(method = 'averaging', kernel = (15, 15))
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
