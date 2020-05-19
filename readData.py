import sys
import os
import cv2
import math
import time
import busio
import board
import numpy as np


import adafruit_amg88xx

# take a folder name passed as arg
if len(sys.argv) < 3:
        print('#usage readData.py folderName numFrames')
        sys.exit()

folder = str(sys.argv[1])
frames = int(sys.argv[2])

# create a folder for new readings

try:
        os.mkdir("./data/" + folder, 0o777)
except OSError:
        print('Creatin of the directory failed (duplicate?)')
        sys.exit()

# initialize sensor
i2c_bus = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_amg88xx.AMG88XX(i2c_bus)


time.sleep(2)

for i in range(frames):
        data = sensor.pixels
        frame = np.asarray(data)
        cv2.imwrite('./data/' + folder + '/pic' +str(i)+ '.png', frame)
        print(i)



