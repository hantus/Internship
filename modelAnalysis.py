import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix , accuracy_score
from sklearn.dummy import DummyClassifier
from joblib import dump, load
import cv2
import sys

mode = None
# if 1 is passed only properly classified data will be shown
# if 0 then miscalssified data will be shown
# therwize all data will be shown
if len(sys.argv) > 1:
    mode = int(sys.argv[1])

def getData(testSize):
    X = np.load('data/preprocessedData/dataSet5.npy')
    # threshold = 1
    # X = (X > threshold).astype(np.int_)
    X = np.reshape(X, (X.shape[0],320))
    # X = np.reshape(X, (X.shape[0],640))

    y = np.load('data/preprocessedData/labels5.npy')
    y = np.reshape(y, (y.shape[0]))

    x_train , x_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=1)
    return x_train , y_train, x_test, y_test


x_train , y_train, x_test, y_test = getData(0.25)

# analysis of the test data
total = y_test.shape[0]
allIN = np.count_nonzero(y_test == 3)
allOUT = np.count_nonzero(y_test == 2)
allX = np.count_nonzero(y_test == 1)
print("Total number of test samples: {}".format(total))
print("Examples of IN {} which is {}% of the data".format(allIN, round(allIN/total*100, 2)))
print("Examples of OUT {} which is {}% of the data".format(allOUT, round(allOUT/total*100, 2)))
print("Examples of IN {} which is {}% of the data".format(allX, round(allX/total*100, 2)))

model = load("data/models/prep5.joblib")

y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy : ", acc)
print("Number of misclassified samples: {}".format(int(total * (1-acc))))

# breakdown of the misclassified
grINprOUT = 0
grINprX = 0
grOUTprIN = 0
grOUTprX = 0
grXprIN = 0
grXprOUT = 0

for i in range(total):
    if y_test[i] != y_pred[i]:
        if (y_test[i] == 3) & (y_pred[i] == 2):
            grINprOUT += 1
        elif (y_test[i] == 3) & (y_pred[i] == 1):
            grINprX += 1
        elif (y_test[i] == 2) & (y_pred[i] == 3):
            grOUTprIN += 1
        elif (y_test[i] == 2) & (y_pred[i] == 1):
            grOUTprX += 1
        elif (y_test[i] == 1) & (y_pred[i] == 3):
            grXprIN += 1
        elif (y_test[i] == 1) & (y_pred[i] == 2):
            grXprOUT += 1
print("Ground truth IN classified as OUT: {}".format(grINprOUT))        
print("Ground truth IN classified as X: {}".format(grINprX))        
print("Ground truth OUT classified as IN: {}".format(grOUTprIN))        
print("Ground truth OUT classified as X: {}".format(grOUTprX))        
print("Ground truth X classified as IN: {}".format(grXprIN))        
print("Ground truth X classified as OUT: {}".format(grXprOUT))        

data = np.reshape(x_test, (len(x_test), 8, 40))


for i in range(data.shape[0]):

    # color for the labels, green if correct prediction, red if incorrect
    color = None
    if y_test[i] == y_pred[i]:
        color = (0,255,0)
        if mode == 0:
            continue
    else:
        color = (0,0,255)
        if mode == 1:
            continue

    cv2.imwrite('data/temp/pic.png', data[i])
    frame = cv2.imread('data/temp/pic.png')
    ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV)
    frame = cv2.resize(frame, (4000, 800), interpolation=cv2.INTER_NEAREST)
    for j in range(1,11):
        start_point = (800 * j , 0)
        end_point = (800 * j , 800)
        frame = cv2.line(frame, start_point, end_point, (0,0,0), 3)
        frame = cv2.line(frame, (800 * j - 400, 0), (800 * j - 400, 800), (0,0,255), 2)



    # print ground truth and predicted label
    cv2.putText(frame, str(y_test[i]) + ' - ' + str(y_pred[i]) , (10, 750), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)

    # print frame number
    cv2.putText(frame, str(i), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.imshow('frame', frame)

    ch = cv2.waitKey()
    if ch == 113:
        break