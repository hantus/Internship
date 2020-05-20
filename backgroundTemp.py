import cv2
import numpy as np
# bck = []
# bck2 = []

# for i in range(1000):
#     frame = cv2.imread('data/background/pic'+ str(i) +'.png', cv2.IMREAD_GRAYSCALE)
#     frame2 = cv2.imread('data/background2/pic'+ str(i) +'.png', cv2.IMREAD_GRAYSCALE)
#     frame = np.asfarray(frame)
#     frame2 = np.asfarray(frame2)
#     bck.append(frame)
#     bck2.append(frame2)

# print("background 1 average {}".format((np.asfarray(bck)).mean()))
# print("background 2 average {}".format((np.asfarray(bck2)).mean()))

readings = np.load('data/2ppl.npy')
print('average {}'.format(np.average(readings)))
print('max {}'.format(np.max(readings)))
print('min {}'.format(np.min(readings)))

