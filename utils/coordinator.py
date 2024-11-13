import numpy as np
import cv2 as cv


# mouse callback function
def draw_lines(event, x, y, flags, param):
    global ix,iy,sx,sy
    global LIST_COORDINATES 
    # if the left mouse button was clicked, record the starting
    if event == cv.EVENT_LBUTTONDOWN:

        # draw circle of 2px
        cv.circle(img, (x, y), 3, (0, 0, 127), -1)

        if ix != -1: # if ix and iy are not first points, then draw a line
            cv.line(img, (ix, iy), (x, y), (0, 0, 127), 2, cv.LINE_AA)
        else: # if ix and iy are first points, store as starting points
            sx, sy = x, y
        ix,iy = x, y
        if [x, y] not in LIST_COORDINATES:
            LIST_COORDINATES.append([x,y])
        
    elif event == cv.EVENT_LBUTTONDBLCLK:
        ix, iy = -1, -1 # reset ix and iy
        if flags == 33: # if alt key is pressed, create line between start and end points to create polygon
            cv.line(img, (x, y), (sx, sy), (0, 0, 127), 2, cv.LINE_AA)



if __name__ == "__main__":
    ix,iy,sx,sy = -1,-1,-1,-1
    LIST_COORDINATES = []
    IMG_PATH = 'libs/drawing/Godzilla.jpeg'

    # Example for running this script:
    # from libs.drawing.coordinates import draw_lines
    
    # read image from path and add callback
    img = cv.resize(cv.imread(IMG_PATH), (1280, 720))
    cv.namedWindow('image') 
    cv.setMouseCallback('image',draw_lines)
    # Drawing by left click (left mouse)
    # On the last point, hold 'alt' and double clicks left mouse,
    #   for draw the line of first point to the last point.

    # ...
    # And then add those lines below into the process scripts
    while(1):
        cv.imshow('image',img)
        if cv.waitKey(20) & 0xFF == 27:
            break

    cv.destroyAllWindows()
