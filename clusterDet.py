# Standard imports
import cv2
import numpy as np
from time import sleep
from sympy.geometry import Point
from sklearn.preprocessing import normalize
import math
import sys


if len(sys.argv) < 2:
    print('#usage clusterDet.py fileName')
    sys.exit()

file = str(sys.argv[1])

class TrackedCluster:

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.frequency = 3
        self.assigned = True


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


def drawLine(img):
    start_point = (200, 0)
    end_point = (200, 400)

    # Green color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.line() method
    # Draw a diagonal green line with thickness of 2 px
    return cv2.line(img, start_point, end_point, color, thickness)


def crossedLine(prevY, currentY, line):
    # if (prev[0] == 0) & (prev[1] == 0):
    #     return 0
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

def clusterDistance(c1, c2):
    return math.sqrt(pow(((c1.x - c2.x)/2), 2) + pow((c1.y - c2.y), 2))

def mergeClusters(c1, c2):
    newCluster = Cluster(c1.id, 2, (c1.x + c2.x)/2, (c1.y + c2.y)/2)
    return newCluster

def clusterData(arr):
    clusters = []
    clusterNo = 2
    size = arr.shape
    for i in range(size[1]):
        for j in range(size[0]):
            if arr[i][j] == 1:
                # check neighbours
                cluster = clusterNo
                arr[i][j] = clusterNo
                
                arr, num, xdim, ydim = checkNeighbors(arr, i, j, clusterNo)
                occurrences = np.count_nonzero(arr == clusterNo)
                clust = Cluster(clusterNo, num, (xdim/occurrences) +0.5, (ydim/occurrences)+0.5)
                clusters.append(clust)
                print("Cluster has {} points and its center is at [{}][{}], occurences {}".format(num, xdim/num, ydim/num, occurrences))
                clusterNo += 1
    # merge nearby clusters
    mergedClusters = []
    while len(clusters) > 1:
        tempCluster = clusters.pop(0)
        merged = 0
        for clust in clusters:
            if clusterDistance(tempCluster, clust) < 3:
                mergedClust = mergeClusters(tempCluster, clust)
                mergedClusters.append(mergedClust)
                clusters.remove(clust)
                merged = 1
        if merged == 0:
            mergedClusters.append(tempCluster)

    if len(clusters) > 0:
        mergedClusters.append(clusters.pop(0))
        

    return arr, mergedClusters


def checkNeighbors(arr, i, j, custNo):
    num = 0
    xdim = 0
    ydim = 0
    
    for x in range(max(0, i-1),min(8, i+2)):
        for y in range(max(0, j-1),min(8, j+2)):
            if arr[x][y] == 1:
                arr[x][y] = custNo
                arr, numR, xdimR, ydimR = checkNeighbors(arr, x, y, custNo)
                num =+ numR
                xdim += xdimR
                ydim += ydimR 
    return arr, num + 1, xdim + i, ydim + j






data = np.load('data/'+file+'.npy')
threshold = 23
data = (data > threshold).astype(np.int_)
frames = data.shape[0]
index = 0
trackedClusters = []
idTracker = IdTracker()
people = 0

for i in range(frames):
    print(index)
    data[i], clusters = clusterData(data[i])


    print(data[i])
    index += 1
    cv2.imwrite('data/temp/pic.png', data[i])
    frame = cv2.imread('data/temp/pic.png')#, cv2.IMREAD_GRAYSCALE)
    ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV)
    frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)
    # add 1 pixel border for blob detection to work at edges
    frame = cv2.copyMakeBorder(
        frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)


    for cl in trackedClusters:
        cl.assigned = False

    
    # track clusters
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

            if nearest != None:
                print("near cluster found")
                people += crossedLine(nearest.y, cl.y, 4)
                nearest.x = cl.x
                nearest.y = cl.y
                nearest.assigned = True
            else:
                print('creating new tracked cluster')
                newTrackedCluster = TrackedCluster(idTracker.getID(), cl.x, cl.y)
                trackedClusters.append(newTrackedCluster)

    for trackedCl in trackedClusters:
        if trackedCl.assigned == False:
            trackedCl.frequency -= 1
            if trackedCl.frequency == 0:
                idTracker.releaseID(trackedCl.id)
                trackedClusters.remove(trackedCl)

    

    cv2.putText(frame, str(people), (20, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
