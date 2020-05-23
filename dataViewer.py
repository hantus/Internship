import cv2
import numpy as np
import time
import sys



if len(sys.argv) < 2:
        print('#usage dataViewer.py fileName')
        sys.exit()

file = str(sys.argv[1])

data = np.load('data/'+file+'.npy')
threshold = 23
data = (data > threshold).astype(np.int_)
print(data.shape)

frames = data.shape[0]

for i in range(frames):
    print(data[i])
    cv2.imwrite('data/temp/pic.png', data[i])
    frame = cv2.imread('data/temp/pic.png' , cv2.IMREAD_GRAYSCALE)
    ret,frame = cv2.threshold(frame,0,255,cv2.THRESH_BINARY_INV)
    frame = cv2.resize(frame, (400,400), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(file, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.05)

   





# video_capture = cv2.VideoCapture('data/'+ folder+'/pic%d.png' )

# # while True:
# for i in range(1000):
#     frame = cv2.imread('data/'+ folder+'/pic'+ str(i) +'.png', cv2.IMREAD_GRAYSCALE)
#     # _, frame = video_capture.read(cv2.IMREAD_GRAYSCALE)
#     f = np.asarray(frame)
#     print(f)
#     ret,frame = cv2.threshold(frame,24,255,cv2.THRESH_BINARY_INV)
#     frame = cv2.resize(frame, (400,400), interpolation=cv2.INTER_CUBIC)

#     cv2.imshow(folder, frame)
#     time.sleep(0.05)
#     cv2.waitKey()


# video_capture.release() #closes the webcam
# cv2.destroyAllWindows()