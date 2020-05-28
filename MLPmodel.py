
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix , accuracy_score
from sklearn.dummy import DummyClassifier
from joblib import dump, load




def getData(testSize):
    X = np.load('data/preprocessedData/1person_add_preprocessed5_merged.npy')
    # the below 2 lines are needed for training on binary data
    # threshold = 23
    # X = (X > threshold).astype(np.int_)

    X = np.reshape(X, (X.shape[0],320))
    # X = np.reshape(X, (X.shape[0],640))
    y = np.load('data/preprocessedData/1person_add_preprocessed5_merged_Labels.npy')
    y = np.reshape(y, (y.shape[0]))

    x_train , x_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=1)
    return x_train , y_train, x_test, y_test

# Building a model


x_train , y_train, x_test, y_test = getData(0.25)
print("x_train {}, y_train {}, x_test {}, y_test {}".format(x_train.shape, y_train.shape,x_test.shape, y_test.shape))

curAcc = 0
for i in range(50):
    model = MLPClassifier(hidden_layer_sizes=(100, 100),activation='relu', solver='adam')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # proba = model.predict_proba(x_test)
    acc = accuracy_score(y_test, y_pred)
    # print("Accuracy : ", acc)
    if acc > curAcc:
        print("Accuracy : ", acc)
        dump(model, 'data/models/prep5.joblib')
        print("Saving a new model")
        curAcc = acc


# compare against dummy model

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train, y_train)
y_pred = dummy_clf.predict(x_test)
print("Dummy Model (most-frequent) Accuracy : ", round(accuracy_score(y_test, y_pred),2))

dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(x_train, y_train)
y_pred = dummy_clf.predict(x_test)
print("Dummy Model (stratified) Accuracy : ", round(accuracy_score(y_test, y_pred),2))

dummy_clf = DummyClassifier(strategy="prior")
dummy_clf.fit(x_train, y_train)
y_pred = dummy_clf.predict(x_test)
print("Dummy Model (prior) Accuracy : ", round(accuracy_score(y_test, y_pred),2))

dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf.fit(x_train, y_train)
y_pred = dummy_clf.predict(x_test)
print("Dummy Model (uniform) Accuracy : ", round(accuracy_score(y_test, y_pred),2))

