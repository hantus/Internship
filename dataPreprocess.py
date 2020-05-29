import numpy as np
import math
import sys


if len(sys.argv) < 2:
    print('#usage dataPreprocess.py fileName')
    sys.exit()


file = str(sys.argv[1])


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
                x = (xdim/occurrences) +0.5
                y = (ydim/occurrences)+0.5
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
    
    # remove the cluster from one side if it is between left and right side
    sides = [0,0,0,0,0,0,0,0,0,0]

    for cl in mergedClusters:
        if cl.side == 'L':
            sides[cl.id] = -1
        else:
           sides[cl.id] = 1 

    for i in range(size[1]):
        for j in range(size[0]):
            if arr[i][j] > 0:
                side = sides[arr[i][j]]
                if (side == -1) & (j > 3):
                    arr[i][j] = 0
                elif (side == 1) & (j < 4):
                    arr[i][j] = 0

        
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





data = np.load('data/'+file+'.npy')
threshold = 23
data = (data > threshold).astype(np.int_)

for i in range(data.shape[0]):
    data[i], cl = clusterData(data[i])



np.save('data/preprocessedData/'+file+'_preprocessed.npy', data)