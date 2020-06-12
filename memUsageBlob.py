from memory_profiler import memory_usage
from memory_profiler import profile
from time import sleep
# import time

@profile (precision=2)
def runProg():
    import numpy as np
    import sys

    data = np.load('data/newData.npy')
    frames = data.shape[0]
    ######################################
    import math
    class TrackedCluster:

        def __init__(self, id, x, y, side):
            self.id = id
            self.x = x
            self.y = y
            self.frequency = 3
            self.assigned = True
            self.side = side

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

    def clusterDistance(c1, c2):
        return math.sqrt(pow(((c1.x - c2.x)/2), 2) + pow((c1.y - c2.y), 2))

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

    def clusterData(arr):
        clusters = []
        clusterID = 2
        size = arr.shape
        for i in range(size[1]):
            for j in range(size[0]):
                if arr[i][j] == 1:
                    cluster = clusterID
                    arr[i][j] = clusterID
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

    threshold = 0
    for i in range(10):
        frame = data[i]
        maxTemp = np.max(frame)
        if maxTemp > threshold:
            threshold = maxTemp

    data = (data > threshold).astype(np.int_)
    trackedClusters = []
    idTracker = IdTracker()
    people = 0
    # start_time = time.time()
    for i in range(frames):

        data[i], clusters = clusterData(data[i])
        for cl in trackedClusters:
            cl.assigned = False
        if len(clusters) > 0:
            for cl in clusters:
                nearest = None
                maxDistance = 2.1
                for trackedCl in trackedClusters:
                    dist = math.sqrt(pow((trackedCl.x - cl.x), 2) + pow(
                            (trackedCl.y - cl.y), 2))
                    if dist < maxDistance:
                        nearest = trackedCl
                        maxDistance = dist
                if nearest != None:
                    side, ppl = crossedLine(nearest, cl.y, 4)
                    people += ppl
                    nearest.side = side
                    nearest.x = cl.x
                    nearest.y = cl.y
                    nearest.assigned = True
                else:
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
        for trackedCl in trackedClusters:
            if trackedCl.assigned == False:
                trackedCl.frequency -= 1
                if trackedCl.frequency == 0:
                    idTracker.releaseID(trackedCl.id)
                    trackedClusters.remove(trackedCl)
            else :
                trackedCl.frequency = 3
    # print("--- %s seconds ---" % (time.time() - start_time))



if __name__ == '__main__':
    runProg()
