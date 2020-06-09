import numpy as np
from collections import deque
import random




dataSet1 = np.load('data/1person.npy')
dataSet2 = np.load('data/1person_hat.npy')
dataSet3 = np.load('data/1person_hood.npy')
dataSet4 = np.load('data/2ppl.npy')
dataSet5 = np.load('data/2ppl_1hat.npy')
dataSet6 = np.load('data/1person_add.npy')
dataSet7 = np.load('data/2ppl_add.npy')

sets = []

sets.append(dataSet1)
sets.append(dataSet2)
sets.append(dataSet3)
sets.append(dataSet4)
sets.append(dataSet5)
sets.append(dataSet6)
sets.append(dataSet7)


labels = np.load('data/labels2.npy')

sequential_data = []
queue = deque(maxlen=10)
label_index = 0 

for set in sets:
    queue.clear()
    for i in range(len(set)):
        queue.append(set[i]/37)

        if len(queue) == 10:
            sequential_data.append([np.array(queue), labels[label_index]])
            label_index += 1
# print(sequential_data[20][0])
random.shuffle(sequential_data)

data = []
labels = []


for dat, gt in sequential_data:
    data.append(dat)
    labels.append(gt)
data = np.asfarray(data)
labels = np.asfarray(labels)

labels[labels == 3] = 0

# FOR A BALLANCE DATA SET (WHERE NUMBER OF ALL CLASSES IS THE SAME) UNCOMMENT THE BELOW CODE
# lower = min(np.count_nonzero(labels == 2), np.count_nonzero(labels == 3))
# print(lower)


# dataSet = []
# groundTruth = []

# one = lower
# two = lower
# three = lower
# for dat, gt in sequential_data:
#     if gt == 1:
#         if one != 0:
#             dataSet.append(dat)
#             groundTruth.append(gt)
#             one -= 1
#     elif gt == 2:
#         if two != 0:
#             dataSet.append(dat)
#             groundTruth.append(gt)
#             two -= 1
#     elif gt == 0:
#         if three != 0:
#             dataSet.append(dat)
#             groundTruth.append(gt)
#             three -= 1

# dataSet = np.asfarray(dataSet)
# groundTruth = np.asfarray(groundTruth)



np.save('data/rnn/dataUnbalanced.npy', data)
np.save('data/rnn/labelsUnbalanced.npy', labels)



#  TO VISUALISE THE RESULTS UNCOMMENT THE BELOW CODE
# import cv2
# numOfFrames = 10
# index = 0
# for data, target in sequential_data:
#     index +=1
    

#     mergedFrame = None

#     mergedFrame = data[0]
#     for a in range(1,10):
#         mergedFrame = np.hstack((mergedFrame, data[a]))
#     threshold = 0.635
#     mergedFrame = (mergedFrame > threshold).astype(np.int_)
#     print(mergedFrame)


#     cv2.imwrite("data/temp/temp.png", mergedFrame)
#     frame = cv2.imread("data/temp/temp.png")
#     ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV)
#     frame = cv2.resize(frame, (800*numOfFrames, 800), interpolation=cv2.INTER_NEAREST)
#     for j in range(1,numOfFrames+1):
#         start_point = (800 * j , 0)
#         end_point = (800 * j , 800)
#         frame = cv2.line(frame, start_point, end_point, (0,0,0), 3)
#         frame = cv2.line(frame, (800 * j - 400, 0), (800 * j - 400, 800), (0,0,255), 2)
#     # cv2.putText(frame, str(i), (10, 750), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
#     label = 'not labeled'
#     if target == 1:
#         label = 'X'
#     elif target == 2:
#         label = 'OUT'
#     elif target == 3:
#         label = 'IN'
#     cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
#     cv2.putText(frame, str(index), (10, 750), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
#     cv2.imshow(label, frame)
#     ch = cv2.waitKey()
#     if ch == 2:
#         labels[i] = 2
#     elif ch == 3:
#         labels[i] = 3
#     elif (ch == 1) | (ch == 0):
#         labels[i] = 1
#     elif ch == 113:
#         break

# cv2.destroyAllWindows()