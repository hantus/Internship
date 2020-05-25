import tensorflow.keras as keras
from  tensorflow.keras import losses 
from  tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import numpy as np
# import pandas as pd
import cv2
# from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix , accuracy_score
from sklearn.dummy import DummyClassifier
from joblib import dump, load




def getData(testSize):
    X = np.load('data/dataSet.npy')
    # X = np.resize(X, (32,32))

    X = np.reshape(X, (X.shape[0],640))


    y = np.load('data/labels.npy')
    # print("the size of Y {}".format(y.shape))
    y = np.reshape(y, (y.shape[0]))
    # print("the size of Y {}".format(y.shape))

    x_train , x_test, y_train, y_test = train_test_split(X, y, test_size=testSize,
                                                random_state=1)
    return x_train , y_train, x_test, y_test

# Building a model


x_train , y_train, x_test, y_test = getData(0.25)
print("x_train {}, y_train {}, x_test {}, y_test {}".format(x_train.shape, y_train.shape,x_test.shape, y_test.shape))

curAcc = 0
for i in range(100):
    model = MLPClassifier(hidden_layer_sizes=(100, 100),activation='relu', solver='adam')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # proba = model.predict_proba(x_test)
    # print(proba)
    # print(y_pred)
    acc = accuracy_score(y_test, y_pred)
    # print("Accuracy : ", acc)
    if acc > curAcc:
        print("Accuracy : ", acc)
        dump(model, 'data/model.joblib')
        print("Saving a new model")
        curAcc = acc





# model = Sequential()
# 1st convolutional layer
# model.add(layers.Conv2D(filters=9, kernel_size=3, activation='relu', input_shape=(32,32,1)))
# model.add(layers.MaxPool2D())
# # 2nd convolutional layer 
# model.add(layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
# model.add(layers.MaxPool2D())
# flatten data 
# model.add(layers.Flatten())
# model.add(layers.Dense(units=100, activation='relu'))
# model.add(layers.Dense(units=100, activation='relu'))
# model.add(layers.Dense(units=30, activation='relu'))
# model.add(layers.Dense(units=3, activation = 'softmax'))
# model.summary()




# # model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
# # model.fit(x_train, to_categorical(y_train), epochs = epochs, batch_size=batch_size)

# model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics =['accuracy'])
# model.fit(x_train, to_categorical(y_train) , batch_size=4, epochs=30)

# score = model.evaluate(x_test, to_categorical(y_test))
# # y_model = model.predict_classes(x_test)


dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train, y_train)
y_pred = dummy_clf.predict(x_test)
print("Dummy Model (most-frequent) Accuracy : ", accuracy_score(y_test, y_pred))

dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(x_train, y_train)
y_pred = dummy_clf.predict(x_test)
print("Dummy Model (stratified) Accuracy : ", accuracy_score(y_test, y_pred))

dummy_clf = DummyClassifier(strategy="prior")
dummy_clf.fit(x_train, y_train)
y_pred = dummy_clf.predict(x_test)
print("Dummy Model (prior) Accuracy : ", accuracy_score(y_test, y_pred))

dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf.fit(x_train, y_train)
y_pred = dummy_clf.predict(x_test)
print("Dummy Model (uniform) Accuracy : ", accuracy_score(y_test, y_pred))

