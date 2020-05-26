
import numpy as np
import cv2
from joblib import  load

# data = np.load('data/1person10_merged.npy')
# model = load("data/models/model1.joblib")
# one = 0
# two = 0
# three = 0
# for i in range(data.shape[0]):
#     frame = data[i]
#     frame = np.reshape(frame,(1,640))
#     pred = model.predict(frame)
#     if pred[0] == 1:
#         one += 1
#     elif pred[0] == 2:
#         two += 1
#     elif pred[0] == 3:
#         three += 1

# print("one {}, two {}, three {}".format(one, two, three))

queue = []

for i in range(20):
    if len(queue) == 10:
        queue.pop(0)

    queue.append(i)
    print(queue)

    



