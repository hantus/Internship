import sys
import numpy as np



if len(sys.argv) < 3:
    print('#usage frameMerger.py fileName numOfFrames')
    sys.exit()



file = str(sys.argv[1])
# how many frames to merge into 1 frame
frameNum = int(sys.argv[2])

data = np.load('data/'+str(file)+'.npy')
frames = data.shape[0]
mergedFrames = []


for i in range(frames - (frameNum -1)):
    mergedFrame = data[i]
    for j in range(i+1, i + frameNum):
        mergedFrame = np.hstack((mergedFrame, data[j]))
    mergedFrames.append(mergedFrame)

mergedFrames = np.asfarray(mergedFrames)

np.save('data/'+str(file)+ str(frameNum)+'_merged.npy', mergedFrames)

