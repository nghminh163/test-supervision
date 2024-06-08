# Import dependencies
import cv2
# Read Images
img = cv2.imread('./0002.jpg')
# Display Image
cv2.imshow('Original Image', img)
cv2.waitKey(0)
# Print error message if image is null
if img is None:
    print('Could not read image')
# Draw line on image
imageLine = img.copy()
pointA = (200, 80)
pointB = (450, 80)
cv2.line(imageLine, pointA, pointB, (255, 255, 0),
         thickness=3, lineType=cv2.LINE_AA)
cv2.imshow('Image Line', imageLine)
cv2.waitKey(0)
