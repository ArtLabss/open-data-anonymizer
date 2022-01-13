# Supplementary functions and variables
import random

def find_middle(x, y, w, h) -> tuple:
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
     

