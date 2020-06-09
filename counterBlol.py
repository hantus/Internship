# Standard imports
import cv2
import numpy as np
from time import sleep
from sympy.geometry import Point
import math
import sys


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

class TrackedCluster:

    def __init__(self, id, x, y, side):
        self.id = id
        self.x = x
        self.y = y
        # allows for a cluster not to be detected in 2 frames in a row
        self.frequency = 3
        self.assigned = True
        self.side = side

# cluster id generator
class IdTracker:
    def __init__(self):
        self.pool = 10
        self.list = [True, True, True, True,
                     True, True, True, True, True, True]

    def getID(self):
        index = self.list.index(True)
        self.list[index] = False
        return index

    def releaseID(self, id):
        self.list[id] = True


# checks if a line was crosses
def crossedLine(prev, currentY, line):

    if (prev.side == 'L') & (currentY > line):
        return 'R', 1
    if (prev.side == 'R') & (currentY < line):
        return 'L', -1
    return prev.side, 0


class Cluster:
    def __init__(self, id, points, x, y, side):
        self.id = id
        self.points = points
        self.x = x
        self.y = y
        self.side = side

# returns distance between 2 clusters (their mid-points)
def clusterDistance(c1, c2):
    return math.sqrt(pow(((c1.x - c2.x)/2), 2) + pow((c1.y - c2.y), 2))


# merges 2 clusters into 1
def mergeClusters(c1, c2):
    totalPoints = c1.points + c2.points
    x = ((c1.x * c1.points) + (c2.x * c2.points))/totalPoints
    y = ((c1.y * c1.points) + (c2.y * c2.points))/totalPoints
    side  = None
    if y <= 4:
        side = 'L'
    else:
        side = 'R'
    newCluster = Cluster(c1.id, totalPoints, x, y, side)
    return newCluster

# performs clustering on an image provided as an np array
# returns 
def clusterData(arr):
    clusters = []
    # starting id
    clusterID = 2
    size = arr.shape
    for i in range(size[1]):
        for j in range(size[0]):
            # if cell = 1 then it is a new cluster
            if arr[i][j] == 1:
                cluster = clusterID
                arr[i][j] = clusterID
                # recursively find all points of the cluster
                arr, xdim, ydim = checkNeighbors(arr, i, j, clusterID)
                occurrences = np.count_nonzero(arr == clusterID)
                x = (xdim/occurrences) + 0.5
                y = (ydim/occurrences)+ 0.5
                side  = None
                if y <= 4:
                    side = 'L'
                else:
                    side = 'R'
                clust = Cluster(clusterID, occurrences, x, y, side)
                clusters.append(clust)
                clusterID += 1
    # merge nearby clusters
    mergedClusters = []
    while len(clusters) > 1:
        tempCluster = clusters.pop(0)
        merged = 0
        for clust in clusters:
            if clusterDistance(tempCluster, clust) < 2:
                mergedClust = mergeClusters(tempCluster, clust)
                arr[arr == clust.id] = tempCluster.id
                mergedClusters.append(mergedClust)
                clusters.remove(clust)
                merged = 1
        if merged == 0:
            mergedClusters.append(tempCluster)

    if len(clusters) > 0:
        mergedClusters.append(clusters.pop(0))
       
    return arr, mergedClusters

# recursively looks for all pixels that belong to the same cluster
def checkNeighbors(arr, i, j, custNo):
    xdim = 0
    ydim = 0
    
    for x in range(max(0, i-1),min(8, i+2)):
        for y in range(max(0, j-1),min(8, j+2)):
            if arr[x][y] == 1:
                arr[x][y] = custNo
                arr, xdimR, ydimR = checkNeighbors(arr, x, y, custNo)
                xdim += xdimR
                ydim += ydimR 
    return arr, xdim + i, ydim + j


# load data file
data = np.load('data/'+file+'.npy')

# binarize the data 
threshold = 0

#set threshold to max environment temp
for i in range(10):
    frame = data[i]
    maxTemp = np.max(frame)
    if maxTemp > threshold:
        threshold = maxTemp

data = (data > threshold).astype(np.int_)
frames = data.shape[0]
trackedClusters = []
idTracker = IdTracker()
people = 0

# count number of enter and exit registered by the program
entering = 0
exiting = 0

for i in range(startFrame, frames):
    data[i], clusters = clusterData(data[i])
    # use cv2 for visualisation
    cv2.imwrite('data/temp/pic.png', data[i])
    frame = cv2.imread('data/temp/pic.png')
    ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV)
    frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)

    
    # add 1 pixel border 
    frame = cv2.copyMakeBorder(
        frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

    # marked all tracked clusters as not assigned
    for cl in trackedClusters:
        cl.assigned = False
    
 # assign found clusters to tracked clusters
    if len(clusters) > 0:
        for cl in clusters:
            # assign the cluster to the nearest tracked cluster if exists
            nearest = None
            maxDistance = 2.1
            for trackedCl in trackedClusters:
                dist = math.sqrt(pow((trackedCl.x - cl.x), 2) + pow(
                        (trackedCl.y - cl.y), 2))
                # print("dist for tracked cl {} is {}".format(trackedCl.id, dist))
                if dist < maxDistance:
                    nearest = trackedCl
                    maxDistance = dist
            # if we found a near cluster assigned it to tracked cluster
            if nearest != None:
                side, ppl = crossedLine(nearest, cl.y, 4)
                if ppl == 1:
                    entering += 1
                elif ppl == -1:
                    exiting += 1
                people += ppl
                nearest.side = side
                nearest.x = cl.x
                nearest.y = cl.y
                nearest.assigned = True
            # else create a new tracked cluster
            else:
                # ingnore clusters that appeared in the middle and are of size 1. They are just noise
                if(cl.points == 1) & (cl.y > 2) & (cl.y < 5):
                    data[i][data[i] == cl.id] = 0
                else:   
                    side = None
                    if cl.y <= 4:
                        side = 'L'
                    else:
                        side = 'R'

                    newTrackedCluster = TrackedCluster(idTracker.getID(), cl.x, cl.y, side)
                    trackedClusters.append(newTrackedCluster)
    # decrease frequency of tracked clusters that were not found and delete if frequency reaches 0
    for trackedCl in trackedClusters:
        if trackedCl.assigned == False:
            trackedCl.frequency -= 1
            if trackedCl.frequency == 0:
                idTracker.releaseID(trackedCl.id)
                trackedClusters.remove(trackedCl)
        else :
            # reset frequency of all assigned clusters to 3 
            trackedCl.frequency = 3

    
    # display number of ppl in the room by cluster detection
    cv2.putText(frame, str(people), (20, 380),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # display frame number
    cv2.putText(frame, str(i), (20, 25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # draw corssing line in the middle
    frame = cv2.line(frame, (200, 0), (200, 400), (0,0,255), 2)

    for cluster in clusters:
        cv2.circle(frame, (int(cluster.y*50), int(cluster.x*50)),20,(0,255,0), -1)
    for item in trackedClusters:
        if item.assigned:
            cv2.putText(frame, str(item.id), (int(item.y*50 - 10), int(item.x*50 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow(file, frame)
    ch = cv2.waitKey()
    if ch == 113:
        break
    sleep(0.05)


cv2.destroyAllWindows()
print('{} - Recorded number of enterings (by clustering algorithm): {}, recorded number of exiting: {}'.format(file, entering, exiting))
