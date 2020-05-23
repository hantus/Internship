import cv2
import numpy as np
# bck = []
# bck2 = []

# for i in range(1000):
#     frame = cv2.imread('data/background/pic'+ str(i) +'.png', cv2.IMREAD_GRAYSCALE)
#     frame2 = cv2.imread('data/background2/pic'+ str(i) +'.png', cv2.IMREAD_GRAYSCALE)
#     frame = np.asfarray(frame)
#     frame2 = np.asfarray(frame2)
#     bck.append(frame)
#     bck2.append(frame2)

# print("background 1 average {}".format((np.asfarray(bck)).mean()))
# print("background 2 average {}".format((np.asfarray(bck2)).mean()))

background = np.load('data/background.npy')
print('Background - average {}, min {}, max {}'.format(np.average(background), np.min(background), np.max(background)))
print()

onePerson = np.load('data/1person.npy')
print('1 person - average {}, min {}, max {}'.format(np.average(onePerson), np.min(onePerson), np.max(onePerson)))
unique = np.unique(onePerson)
print(unique[unique > np.max(background)])
print()

onePersonHat = np.load('data/1person_hat.npy')
print('1 person + hat - average {}, min {}, max {}'.format(np.average(onePersonHat), np.min(onePersonHat), np.max(onePersonHat)))
unique = np.unique(onePersonHat)
print(unique[unique > np.max(background)])
print()

onePersonHood = np.load('data/1person_hood.npy')
print('1 person + hood - average {}, min {}, max {}'.format(np.average(onePersonHood), np.min(onePersonHood), np.max(onePersonHood)))
unique = np.unique(onePersonHood)
print(unique[unique > np.max(background)])
print()

twoPpl = np.load('data/2ppl.npy')
print('2 ppl - average {}, min {}, max {}'.format(np.average(twoPpl), np.min(twoPpl), np.max(twoPpl)))
unique = np.unique(twoPpl)
print(unique[unique > np.max(background)])
print()

twoPplHat = np.load('data/2ppl.npy')
print('2 ppl + 1 hat - average {}, min {}, max {}'.format(np.average(twoPplHat), np.min(twoPplHat), np.max(twoPplHat)))
unique = np.unique(twoPplHat)
print(unique[unique > np.max(background)])
print()




