# Standard imports
import cv2
import numpy as np
from time import sleep
import math
import sys
from joblib import load
from sys import getsizeof
from collections import deque




if len(sys.argv) < 2:
    print('#usage clusterDet.py fileName')
    sys.exit()
print("You can exit the application at any time by pressing q")

file = str(sys.argv[1])
# to indicate where you want to start 
startFrame = 0
try:
    startFrame = int(sys.argv[2])
except: 
    print("no starting frame provided")



# load data file
data = np.load('data/'+file+'.npy')
# get number of frames in the file
frames = data.shape[0]
# variable to keep track of people in the room
people = 0

# load neural model
model = load("data/models/model-70-100.joblib")
# queue to 
queue = deque(maxlen=10)
nnPeople = 0


# count number of enter and exit
entering = 0
exiting = 0

for i in range(startFrame, frames):

    cv2.imwrite('data/temp/pic.png', data[i])
    frame = cv2.imread('data/temp/pic.png')
    print(f"the size of frame is {getsizeof(frame)}")
    ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV)
    frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)

    # append the new frame
    queue.append(data[i])
    # if we have 10 frames join them toheter and get a prediction from the model
    if len(queue) == 10:
        # merge the 10 frames
        queue2 = np.asarray(queue)
        mergedArray = queue2[0]

        for k in range(1, 10):
            mergedArray = np.hstack((mergedArray,  queue2[k]))

        mergedArray = np.reshape(mergedArray, (1,640))
        pred = model.predict(mergedArray)
        # if pred equals 3 somebody entered, if 2 somebody left, if 1 no action
        if pred[0] == 3:
            nnPeople += 1
            queue.clear()
        elif pred[0] == 2:
            nnPeople -= 1
            queue.clear()


    # add 1 pixel border 
    frame = cv2.copyMakeBorder(
        frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
 
    
    # display number of ppl in the room by cluster detection
    cv2.putText(frame, str(people), (20, 380),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # display frame number
    cv2.putText(frame, str(i), (20, 25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # draw corssing line in the middle
    frame = cv2.line(frame, (200, 0), (200, 400), (0,0,255), 2)

    cv2.imshow(file, frame)
    ch = cv2.waitKey()
    if ch == 113:
        break
    sleep(0.05)


cv2.destroyAllWindows()
print('{} - Recorded number of enterings (by clustering algorithm): {}, recorded number of exiting: {}'.format(file, entering, exiting))

