import numpy as np
import cv2
import sys

if len(sys.argv) < 2:
    print('#usage mergedFrameLabeler.py fileName ')
    sys.exit()

file = str(sys.argv[1])
data = np.load('data/'+str(file)+'.npy')
frames = data.shape[0]

try:
    labels = np.load('data/'+str(file)+'_Labels.npy')
    print("label file uploaded")
except:
    print("no label file")
threshold = 23
data = (data > threshold).astype(np.int_)

for i in range(frames):
    cv2.imwrite("data/temp/temp.png", data[i])
    frame = cv2.imread("data/temp/temp.png")
    et, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV)
    frame = cv2.resize(frame, (8000, 800), interpolation=cv2.INTER_NEAREST)
    for i in range(1,11):
        start_point = (800 * i , 0)
        end_point = (800 * i , 800)
        frame = cv2.line(frame, start_point, end_point, (0,0,0), 3)
        frame = cv2.line(frame, (800 * i - 400, 0), (800 * i - 400, 800), (0,0,255), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey()


