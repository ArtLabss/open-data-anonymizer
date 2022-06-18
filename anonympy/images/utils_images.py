# Supplementary functions and variables
import random
import numpy as np
import cv2


def find_middle(x, y, w, h) -> tuple:
    """
    Function for finding the center of a rectangle
    The center of rectangle is the midpoint of the diagonal end points of
     rectangle.
    """
    return int(x + w / 2), int(y + h / 2)


def find_radius(x, y, w, h) -> tuple:
    """
    Function finds the distance between the center and side edge
    """
    side_middle = x + w, (y + y + h) / 2
    center = find_middle(x, y, w, h)
    return side_middle[0] - center[0]


def sap_noise(frame, seed=None):
    random.seed(seed)
    img = frame.copy()
    # Getting the dimensions of the image
    row, col, _ = img.shape
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(8000, 15000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)
        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)
        # Color that pixel to white
        img[y_coord][x_coord] = 255
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(8000, 15000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)
        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)
        # Color that pixel to black
        img[y_coord][x_coord] = 0
    return img


def pixelated(image, blocks=20):
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
            cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)

    return image


def resize(self, new_width=500):
    height, width, _ = self.frame.shape
    ratio = height / width
    new_height = int(ratio * new_width)
    return cv2.resize(self.frame, (new_width, new_height))
