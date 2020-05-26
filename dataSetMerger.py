import numpy as np

dataSet1 = np.load('data/1person10_merged.npy')
dataSet2 = np.load('data/1person_hat10_merged.npy')
dataSet3 = np.load('data/1person_hood10_merged.npy')
dataSet4 = np.load('data/2ppl10_merged.npy')
dataSet5 = np.load('data/2ppl_1hat10_merged.npy')

dataSets = []
dataSets.append(dataSet1)
dataSets.append(dataSet2)
dataSets.append(dataSet3)
dataSets.append(dataSet4)
dataSets.append(dataSet5)

data = []
for ds in dataSets:
    for i in range(ds.shape[0]):
        data.append(ds[i])
data = np.asarray(data)
print(data.shape)

labels1 = np.load('data/1person10_merged_Labels.npy')
labels2 = np.load('data/1person_hat10_merged_Labels.npy')
labels3 = np.load('data/1person_hood10_merged_Labels.npy')
labels4 = np.load('data/2ppl10_merged_Labels.npy')
labels5 = np.load('data/2ppl_1hat10_merged_Labels.npy')

labelSets = []
labelSets.append(labels1)
labelSets.append(labels2)
labelSets.append(labels3)
labelSets.append(labels4)
labelSets.append(labels5)

labels = []
for ds in labelSets:
    for i in range(ds.shape[0]):
        labels.append(ds[i])
labels = np.asarray(labels)
print(labels.shape)

np.save('data/dataSet.npy', data)
np.save('data/labels.npy', labels)

total = labels.shape[0]
allIN = np.count_nonzero(labels == 3)
allOUT = np.count_nonzero(labels == 2)
allX = np.count_nonzero(labels == 1)
print("Total number of samples: {}".format(total))
print("Examples of IN {} which is {}% of the data".format(allIN, round(allIN/total*100, 2)))
print("Examples of OUT {} which is {}% of the data".format(allOUT, round(allOUT/total*100, 2)))
print("Examples of X {} which is {}% of the data".format(allX, round(allX/total*100, 2)))



