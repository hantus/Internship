from memory_profiler import memory_usage
from memory_profiler import profile
from time import sleep


@profile (precision=2)
def runProg():

    import numpy as np
    import sys

    rawData = np.load('data/newData.npy')
    frames = rawData.shape[0]
    #######################################
    from tensorflow import keras
    from collections import deque

    rawData = rawData/37
    model = keras.models.load_model('data/models/rnn/100per')
    queue = deque(maxlen=10)
    nnPeople = 0
    for i in range(frames):

        queue.append(rawData[i])
        if len(queue) == 10:
            queue2 = np.asarray(queue)
            mergedArray = np.reshape(queue2, (1,10,64))
            proba = model.predict(mergedArray)
            probaLabel = proba[0]
            pred = np.argmax(proba)
            if pred == 0:
                nnPeople += 1
                queue.clear()
            elif pred == 2:
                nnPeople -= 1
                queue.clear()

if __name__ == '__main__':
    runProg()