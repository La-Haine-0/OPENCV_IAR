import cv2
import numpy as np
from matplotlib import pyplot as plt
 
 
img = cv2.imread('bk4.jpg')
rows, cols, ch = img.shape
 
pts1 = np.float32([[300, 200],
                   [300, 550],
                   [700, 600]])
 
pts2 = np.float32([[300, 500],
                   [300, 750],
                   [700, 600]])
 
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
 
plt.subplot(121)
plt.imshow(img)
plt.title('Input')
 
plt.subplot(122)
plt.imshow(dst)
plt.title('Output')
 
plt.show()
 
# Displaying the image
while(1):
     
    cv2.imshow('image', img)
    cv2.imwrite("image-aff", dst)
    if cv2.waitKey(20) & 0xFF == 27:
        break
         
cv2.destroyAllWindows()