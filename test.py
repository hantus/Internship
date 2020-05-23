
import numpy as np
import cv2

data = np.load('data/1person.npy')


merged = data[25]
for i in range(26, 36):
    merged = np.hstack((merged, data[i]))

threshold = 23
merged = (merged > threshold).astype(np.int_)
print(merged)
cv2.imwrite('data/temp/pic.png', merged)
frame = cv2.imread('data/temp/pic.png')
ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('frame', frame)
cv2.waitKey()