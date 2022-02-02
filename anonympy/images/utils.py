# Supplementary functions and variables
import random
import numpy as np
import cv2

def find_middle(x, y, w, h) -> tuple:
        '''
        Supple
        '''
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        m1, m2 = int((x1 + x2)/2), int((y1 + y2)/2)
        return m1, m2


def find_radius(x, y, w, h) -> tuple:
        pt1 = (x, y)
        pt2 = (x+w, y+h)

        side_middle = x + w, (y + y + h) / 2
        center = find_middle(x, y, w, h)
        dis = side_middle[0] - center[0]

        return dis


def sap_noise(frame):
    img = frame.copy()
    # Getting the dimensions of the image
    row , col, _ = img.shape
     
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(5000, 10000)
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
    number_of_pixels = random.randint(5000 , 10000)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to black
        img[y_coord][x_coord] = 0
         
    return img
     

def pixelated(image,  blocks = 20):
          (h, w) = image.shape[:2]
          xSteps = np.linspace(0, w, blocks + 1, dtype="int")
          ySteps = np.linspace(0, h, blocks + 1, dtype="int")

          for i in range(1, len(ySteps)):
               for j in range(1, len(xSteps)):
                    # compute the starting and ending (x, y)-coordinates
                    # for the current block
                    startX = xSteps[j - 1]
                    startY = ySteps[i - 1]
                    endX = xSteps[j]
                    endY = ySteps[i]
                    # extract the ROI using NumPy array slicing, compute the
                    # mean of the ROI, and then draw a rectangle with the
                    # mean RGB values over the ROI in the original image
                    roi = image[startY:endY, startX:endX]
                    (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                    cv2.rectangle(image, (startX, startY), (endX, endY),(B, G, R), -1)

          return image


def resize(self, new_width=500):
        height, width, _ = self.frame.shape
        ratio = height / width
        new_height = int(ratio * new_width)
        return cv2.resize(self.frame, (new_width, new_height))
