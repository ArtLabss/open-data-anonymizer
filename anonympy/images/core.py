import cv2
import numpy as np


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
          self.frame = frame


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

          for face in detections:
               x,y,w,h = face

               self.frame[y:y+h,x:x+w] = cv2.GaussianBlur(self.frame[y:y+h,x:x+w], kernel, cv2.BORDER_DEFAULT)

               if box:
                    cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)


     def imshow(self, fname: str):
          '''
          '''

          cv2.imshow(fname, self.frame)
