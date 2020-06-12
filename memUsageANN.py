from memory_profiler import memory_usage
from memory_profiler import profile
from time import sleep
# import time

@profile (precision=2)
def runProg():

    import numpy as np
    import sys

    rawData = np.load('data/newData.npy')
    frames = rawData.shape[0]

    #########################################
    from joblib import load
    from collections import deque

    model = load("data/models/model-70-100.joblib")
    queue = []
    nnPeople = 0
    # start_time = time.time()
    for i in range(frames):

        queue.append(rawData[i])
        if len(queue) == 10:
            queue2 = np.asarray(queue)
            mergedArray = queue2[0]

            for k in range(1, 10):
                mergedArray = np.hstack((mergedArray,  queue2[k]))

            mergedArray = np.reshape(mergedArray, (1,640))
            pred = model.predict(mergedArray)
            if pred[0] == 3:
                nnPeople += 1
                queue.clear()
            elif pred[0] == 2:
                nnPeople -= 1
                queue.clear()
    # print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    runProg()