import sys
import os
import cv2
import math
import time
# import busio
# import board
import numpy as np


# import adafruit_amg88xx

# take a folder name passed as arg
if len(sys.argv) < 3:
        print('#usage readData.py folderName numFrames')
        sys.exit()

folder = str(sys.argv[1])
frames = int(sys.argv[2])



# initialize sensor
# i2c_bus = busio.I2C(board.SCL, board.SDA)
# sensor = adafruit_amg88xx.AMG88XX(i2c_bus)


time.sleep(2)
readings = []
for i in range(frames):
        # data = sensor.pixels
        # frame = np.asfarray(data)
        a = np.random.rand(2,2)
        readings.append(a)

        print(i)

# convert into np array
readings = np.asfarray(readings)
print(readings)
np.save('data/'+ folder, readings)


