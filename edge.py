import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('im.jpg',0)

# Edge Operators
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
sobel = sobelx + sobely
laplacian = cv2.Laplacian(img,cv2.CV_64F)
canny = cv2.Canny(img,100,200)

cv2.imwrite('Sobelx.jpg',sobelx)
cv2.imwrite('Sobely.jpg',sobely)
cv2.imwrite('Sobel.jpg',sobel)
cv2.imwrite('laplacian.jpg',laplacian)
cv2.imwrite('Canny.jpg',canny)

plt.subplot(321),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobelx Image'), plt.xticks([]), plt.yticks([])
plt.subplot(323),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobely Image'), plt.xticks([]), plt.yticks([])
plt.subplot(324),plt.imshow(sobel,cmap = 'gray')
plt.title('Sobel Image'), plt.xticks([]), plt.yticks([])
plt.subplot(325),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian Image'), plt.xticks([]), plt.yticks([])
plt.subplot(326),plt.imshow(canny,cmap = 'gray')
plt.title('Canny Image'), plt.xticks([]), plt.yticks([])

plt.show()


