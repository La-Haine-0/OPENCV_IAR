from ast import Break
from operator import lt
import cv2
import numpy as np
 
# Read source image.
im_src = cv2.imread('bk1.jpg')
# Four corners of the book in source image
pts_src = np.float32([[726, 91], [638, 634], [324, 237], [968, 310]])
 
# Read destination image.
im_dst = cv2.imread('bk3.jpg')
# Four corners of the book in destination image.
pts_dst = np.float32([[930, 149], [302, 432], [360, 83], [1026, 450]])


# Calculate the homography
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
 
# Warp source image to destination
im_out = cv2.warpPerspective ( im_src, M, (im_src.shape[1], im_src.shape[0]))
 
# Show output
cv2.imshow("Image", im_out)
cv2.imwrite("new-viev.jpg", im_out)
 
cv2.waitKey(0)