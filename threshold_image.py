import cv2
import numpy as np

# Load image, create blank mask, convert to grayscale, Gaussian blur
# then adaptive threshold to obtain a binary image
image = cv2.imread('/Users/vutu/Documents/Projects/ChuNom/sources/chunom/resources/newdb/train/4.jpg')
mask = np.zeros(image.shape, dtype=np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)

# Create horizontal kernel then dilate to connect text contours
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,2))
dilate = cv2.dilate(thresh, kernel, iterations=2)

# Find contours and filter out noise using contour approximation and area filtering
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    x,y,w,h = cv2.boundingRect(c)
    area = w * h
    ar = w / float(h)
    if area > 1200 and area < 50000 and ar < 6:
        cv2.drawContours(mask, [c], -1, (255,255,255), -1)

# Bitwise-and input image and mask to get result
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
result = cv2.bitwise_and(image, image, mask=mask)
result[mask==0] = (255,255,255) # Color background white

cv2.imshow('thresh', thresh)
cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey()