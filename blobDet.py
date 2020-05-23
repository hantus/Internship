# Standard imports
import cv2
import numpy as np
from time import sleep
from sympy.geometry import Point
from sklearn.preprocessing import normalize
import math
import sys


class TrackedBlob:

    def __init__(self, id, position):
        self.id = id
        self.position = position
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



def create_blob_detector(roi_size=(300, 300), blob_min_area=2,  # for 8x8 roi_size=(8, 8), blob_min_area=4
                         blob_min_int=0, blob_max_int=.3, blob_th_step=1):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.filterByColor = True
    params.blobColor = 0
    params.minArea = blob_min_area
    # params.maxArea = roi_size[0]*roi_size[1]
    params.maxArea = 250000
    # params.filterByCircularity = False
    # params.minCircularity = 0.6
    # params.maxCircularity = 1
    params.filterByConvexity = False
    params.filterByInertia = False
    # blob detection only works with "uint8" images.
    params.minThreshold = int(blob_min_int*255)
    params.maxThreshold = int(blob_max_int*255)
    params.thresholdStep = blob_th_step
    # merging
    params.minDistBetweenBlobs = 50
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)


# Set up the detector with default parameters.
detector = create_blob_detector()

x_min = 1000
x_max = 0
people = 0
max_dist = 150

prev = Point(0, 0)


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


def crossedLine(prev, current, line):
    if (prev[0] == 0) & (prev[1] == 0):
        return 0
    if (prev[0]> line) & (current[0] < line):
        return -1
    if (prev[0] < line) & (current[0] > line):
        return 1
    return 0


def withinDist(prev, current, max_dist):
    d = math.sqrt(pow((current.x - prev.x), 2) + pow((current.y - prev.y), 2))
    print(d)
    if d < max_dist:
        return True
    return False


if len(sys.argv) < 2:
    print('#usage dataViewer.py fileName')
    sys.exit()

file = str(sys.argv[1])

data = np.load('data/'+file+'.npy')
threshold = 23
data = (data > threshold).astype(np.int_)
frames = data.shape[0]
index = 0
trackedBlobs = []
idTracker = IdTracker()

for i in range(frames):
    print(index)
    index += 1
    cv2.imwrite('data/temp/pic.png', data[i])
    frame = cv2.imread('data/temp/pic.png', cv2.IMREAD_GRAYSCALE)
    ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV)
    frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)
    # add 1 pixel border for blob detection to work at edges
    frame = cv2.copyMakeBorder(
        frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

    keypoints = detector.detect(frame)

    for blob in trackedBlobs:
            blob.assigned = False

    if keypoints:

        for keypoint in keypoints:
            print("number of blobs detected : {}".format(len(keypoints)))
            print("I am a blob at location x = {}, y = {}".format(keypoint.pt[0], keypoint.pt[1]))
            # assign this blob to the nearest tracked blob
            nearest = None
            distance = 150
            for trackedBlob in trackedBlobs:
                dist = math.sqrt(pow((trackedBlob.position[0] - keypoint.pt[0]), 2) + pow(
                    (trackedBlob.position[1] - keypoint.pt[1]), 2))
                if dist < distance:
                    nearest = trackedBlob
            if nearest != None:
                print("near blob found")
                people += crossedLine(nearest.position, keypoint.pt, 200)
                nearest.position = keypoint.pt
                nearest.assigned = True
            else:
                print("creating new blob")
                # if no blobs within this distance create a new tracked blob
                newTrackedBlob = TrackedBlob(idTracker.getID(), keypoint.pt)
                trackedBlobs.append(newTrackedBlob)

    for item in trackedBlobs:
        if item.assigned == False:
            item.frequency -= 1
            if item.frequency == 0:
                idTracker.releaseID(item.id)
                trackedBlobs.remove(item)



    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
        []), (0, 200, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints = drawLine(im_with_keypoints)
    cv2.putText(im_with_keypoints, str(people), (20, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for item in trackedBlobs:
        if item.assigned:
            cv2.putText(im_with_keypoints, str(item.id), (int(item.position[0]), int(item.position[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # if keypoints:
    #     cv2.circle(im_with_keypoints, ((int)(keypoints[0].pt[0]),(int)(keypoints[0].pt[1])), 2, (0, 255, 0), -1)


    cv2.imshow(file, im_with_keypoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey()
    sleep(0.05)


cv2.destroyAllWindows()
