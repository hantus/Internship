# Standard imports
import cv2
import numpy as np
from time import sleep
from sympy.geometry import Point
import math
import sys
import busio
import board
import adafruit_amg88xx
from joblib import load
from collections import deque



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

# returns an image with line drawn onto it
def drawLine(img):
    start_point = (200, 0)
    end_point = (200, 400)
    color = (0, 0, 255)
    thickness = 2
    return cv2.line(img, start_point, end_point, color, thickness)

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


    # delete small clusters at the edge
    clCopy = np.copy(mergedClusters)
    for cl in clCopy:
        if (cl.y < 0.85) | (cl.y > 7.0):
            arr[arr == cl.id] = 0
            mergedClusters.remove(cl)
        
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

# Initialize senson

i2c_bus = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_amg88xx.AMG88XX(i2c_bus)
sleep(2)

# Get max temp of background

threshold = 0
bckgrd = []
for i in range(15):
    data = sensor.pixels
    frame = np.asfarray(data)
    maxTemp = np.max(frame)
    if maxTemp > threshold:
        threshold = maxTemp

# binarize the data 

data = (data > threshold).astype(np.int_)


trackedClusters = []
idTracker = IdTracker()
people = 0


# load neural model
model = load("data/models/model-70-100.joblib")
queue = deque(maxlen=10)
nnPeople = 0


try:

    while True:
    
        data = sensor.pixels
        data = np.asfarray(data)
        data = (data > threshold).astype(np.int_)
        frame, clusters = clusterData(data)

        # append the new frame
        queue.append(frame)
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


        # marked all tracked clusters as not assigned
        for cl in trackedClusters:
            cl.assigned = False

        # assign found clusters to tracked clusters
        if len(clusters) > 0:
            for cl in clusters:
                # assign the cluster to the nearest tracked cluster if exists
                nearest = None
                distance = 2.1
                for trackedCl in trackedClusters:
                    dist = math.sqrt(pow((trackedCl.x - cl.x), 2) + pow(
                            (trackedCl.y - cl.y), 2))
                    # print("dist for tracked cl {} is {}".format(trackedCl.id, dist))
                    if dist < distance:
                        nearest = trackedCl
                        distance = dist
                # if we found a near cluster assigned it to tracked cluster
                if nearest != None:
                    side, ppl = crossedLine(nearest, cl.y, 4)
                    people += ppl
                    nearest.side = side
                    nearest.x = cl.x
                    nearest.y = cl.y
                    nearest.assigned = True
                # else create a new tracked cluster
                else:
                    # ingnore clusters that appeared in the middle and are of size 1. They are just noise
                    if(cl.points == 1) & (cl.y > 2) & (cl.y < 5):
                        frame[frame == cl.id] = 0
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
except KeyboardInterrupt:
    print("Exit program")



