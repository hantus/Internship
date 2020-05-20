import cv2
import numpy as np
import time
import sys

if len(sys.argv) < 2:
        print('#usage dataViewer.py folderName')
        sys.exit()

folder = str(sys.argv[1])

video_capture = cv2.VideoCapture('data/'+ folder+'/pic%d.png' )

# while True:
for i in range(1000):
    frame = cv2.imread('data/'+ folder+'/pic'+ str(i) +'.png', cv2.IMREAD_GRAYSCALE)
    # _, frame = video_capture.read(cv2.IMREAD_GRAYSCALE)
    f = np.asarray(frame)
    print(f)
    ret,frame = cv2.threshold(frame,24,255,cv2.THRESH_BINARY_INV)
    frame = cv2.resize(frame, (400,400), interpolation=cv2.INTER_CUBIC)

    cv2.imshow(folder, frame)
    time.sleep(0.05)
    cv2.waitKey()


video_capture.release() #closes the webcam
cv2.destroyAllWindows()