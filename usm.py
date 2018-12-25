import cv2
import numpy as np

# Edge Sharpening with Unsharp Masking Technique

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
	# credit: https://github.com/soroushj/python-opencv-numpy-example/blob/master/unsharpmask.py
	
image = cv2.imread('im.jpg',0)
im_unsharped_1 = unsharp_mask(image,sigma=1.0)
im_unsharped_50 = unsharp_mask(image,sigma=50.0)
cv2.imwrite('USM1.jpg',im_unsharped_1)
cv2.imwrite('USM50.jpg',im_unsharped_50)
compare = np.hstack((image,im_unsharped_1,im_unsharped_50))
cv2.imshow('USM',compare)
cv2.waitKey(0)
cv2.destroyAllWindows()