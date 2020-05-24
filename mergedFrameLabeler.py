import numpy as np
import cv2
import sys

if len(sys.argv) < 2:
    print('#usage mergedFrameLabeler.py fileName ')
    sys.exit()
#
file = str(sys.argv[1])
data = np.load('data/'+str(file)+'.npy')
frames = data.shape[0]
labels = None
# used to select a frame number if a file already exists
newFile = 0

# upload the label file of create a new one if it doesn't exist
try:
    labels = np.load('data/'+str(file)+'_Labels.npy')
    print("label file uploaded")
except:
    print("no label file, creating array")
    labels = np.zeros((frames, 1), dtype=int)
    newFile = 1

# keeps the number of a first unlabeled frame
firstEmpty = ((np.where(labels == 0))[0][0])
start = 0

# when resuming previously started labeling of a file
if newFile == 0:
    proceed = 0
    while proceed == 0:
        print("The last labeled frame is {}, do you want to start from the next frame? [y/n]".format(firstEmpty-1))
        response = str(input()) 
        if (response == 'y') | (response == 'Y'):
            print("you selected yes")
            proceed = 1
            start = firstEmpty
        elif (response == 'n') | (response == 'N'):
            print("Please provide frame number wher you would like to start:")
            response = int(input())
            print(response)
            start = response
            proceed = 1

# binarise the images 
threshold = 23
data = (data > threshold).astype(np.int_)

for i in range(start, frames):
    cv2.imwrite("data/temp/temp.png", data[i])
    frame = cv2.imread("data/temp/temp.png")
    et, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV)
    frame = cv2.resize(frame, (8000, 800), interpolation=cv2.INTER_NEAREST)
    for j in range(1,11):
        start_point = (800 * j , 0)
        end_point = (800 * j , 800)
        frame = cv2.line(frame, start_point, end_point, (0,0,0), 3)
        frame = cv2.line(frame, (800 * j - 400, 0), (800 * j - 400, 800), (0,0,255), 2)
    cv2.putText(frame, str(i), (10, 750), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
    label = 'not labeled'
    if labels[i] == 1:
        label = 'X'
    elif labels[i] == 2:
        label = 'OUT'
    elif labels[i] == 3:
        label = 'IN'
    cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.putText(frame, str(i), (10, 750), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
    cv2.imshow(file, frame)
    ch = cv2.waitKey()
    if ch == 2:
        labels[i] = 2
    elif ch == 3:
        labels[i] = 3
    elif (ch == 1) | (ch == 0):
        labels[i] = 1
    elif ch == 113:
        print(labels)
        labels = np.save('data/'+str(file)+'_Labels.npy', labels)
        break


cv2.destroyAllWindows()