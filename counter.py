# Standard imports
import cv2
import numpy as np
from time import sleep
from sympy.geometry import Point
# from sklearn.preprocessing import normalize
import math
import sys
from joblib import load


if len(sys.argv) < 2:
    print('#usage clusterDet.py fileName')
    sys.exit()

file = str(sys.argv[1])

class TrackedCluster:

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        # allows for a cluster not to be detected in 2 frames in a row
        self.frequency = 3
        self.assigned = True

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

# returns an image with line drawn onto it
def drawLine(img):
    start_point = (200, 0)
    end_point = (200, 400)
    color = (0, 0, 255)
    thickness = 2
    return cv2.line(img, start_point, end_point, color, thickness)

# checks if a line was crosses
def crossedLine(prevY, currentY, line):
    if (prevY >= line) & (currentY < line):
        return -1
    if (prevY <= line) & (currentY > line):
        return 1
    return 0


class Cluster:
    def __init__(self, id, points, x, y):
        self.id = id
        self.points = points
        self.x = x
        self.y = y

# returns distance between 2 clusters (their mid-points)
def clusterDistance(c1, c2):
    return math.sqrt(pow(((c1.x - c2.x)/2), 2) + pow((c1.y - c2.y), 2))

# merges 2 clusters into 1
def mergeClusters(c1, c2):
    newCluster = Cluster(c1.id, 2, (c1.x + c2.x)/2, (c1.y + c2.y)/2)
    return newCluster

# performs clustering on an image provided as an np array
# returns 
def clusterData(arr):
    clusters = []
    clusterID = 2
    size = arr.shape
    for i in range(size[1]):
        for j in range(size[0]):
            if arr[i][j] == 1:
                cluster = clusterID
                arr[i][j] = clusterID
                # recursively find all points of the cluster
                arr, xdim, ydim = checkNeighbors(arr, i, j, clusterID)
                occurrences = np.count_nonzero(arr == clusterID)
                clust = Cluster(clusterID, occurrences, (xdim/occurrences) +0.5, (ydim/occurrences)+0.5)
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
threshold = 23.5
data = (data > threshold).astype(np.int_)
frames = data.shape[0]
trackedClusters = []
idTracker = IdTracker()
people = 0

# load neural model
model = load("data/models/modelBin.joblib")
queue = []
nnPeople = 0

for i in range(1100,frames):
    nnData = np.copy(data[i])
    data[i], clusters = clusterData(data[i])

    cv2.imwrite('data/temp/pic.png', data[i])
    frame = cv2.imread('data/temp/pic.png')
    ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV)
    frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)

    # if our queue already has 10 items pop the oldest sub-frame
    if len(queue) == 10:
        queue.pop(0)
    # append the new frame
    queue.append(nnData)
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

    # marked all tracked clusters as not assigned
    for cl in trackedClusters:
        cl.assigned = False
    
    # assign found clusters to tracked clusters
    if len(clusters) > 0:
        for cl in clusters:
            # assign the cluster to the nearest tracked cluster if exists
            nearest = None
            distance = 2
            for trackedCl in trackedClusters:
                dist = math.sqrt(pow((trackedCl.x - cl.x), 2) + pow(
                        (trackedCl.y - cl.y), 2))
                if dist < distance:
                    nearest = trackedCl
            # if we found a near cluster assigned it to tracked cluster
            if nearest != None:
                people += crossedLine(nearest.y, cl.y, 4)
                nearest.x = cl.x
                nearest.y = cl.y
                nearest.assigned = True
            # else create a new tracked cluster
            else:
                newTrackedCluster = TrackedCluster(idTracker.getID(), cl.x, cl.y)
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
    # display number of ppl in the room by neural model
    cv2.putText(frame, str(nnPeople), (80, 380),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 255), 2)
    # display frame number
    cv2.putText(frame, str(i), (20, 25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    frame = drawLine(frame)

    for cluster in clusters:
        cv2.circle(frame, (int(cluster.y*50), int(cluster.x*50)),20,(0,255,0), -1)
    for item in trackedClusters:
        if item.assigned:
            cv2.putText(frame, str(item.id), (int(item.y*50 - 10), int(item.x*50 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow(file, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey()
    sleep(0.05)


cv2.destroyAllWindows()
