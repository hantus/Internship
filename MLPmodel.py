
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix , accuracy_score
from sklearn.dummy import DummyClassifier
from joblib import dump, load




def getData(testSize):
    X = np.load('data/dataSet2.npy')
    # the below 2 lines are needed for training on binary data
    # threshold = 23
    # X = (X > threshold).astype(np.int_)

    # X = np.reshape(X, (X.shape[0],320)) # 320 for 5 merged frames, 640 for 10
    X = np.reshape(X, (X.shape[0],640))
    y = np.load('data/labels2.npy')
    y = np.reshape(y, (y.shape[0]))

    x_train , x_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=1)
    return x_train , y_train, x_test, y_test

# Building a model


x_train , y_train, x_test, y_test = getData(0.25)
print("x_train {}, y_train {}, x_test {}, y_test {}".format(x_train.shape, y_train.shape,x_test.shape, y_test.shape))

curAcc = 0.9725776965265083
l1 = [50,60,70,80,90,100]
l2 = [60,70,80,90,100]
l3 = [60,70,80,90,100]
for layer1 in l1:
    for layer2 in l2:
        for layer3 in l3:
            for i in range(20):
                if i == 0:
                    print("Training model with layers {} - {} - {}".format(layer1, layer2, layer3))
                model = MLPClassifier(hidden_layer_sizes=(layer1, layer2, layer3),activation='relu', solver='adam')
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                # proba = model.predict_proba(x_test)
                acc = accuracy_score(y_test, y_pred)
                print("Accuracy : ", acc)
                if acc > curAcc:
                    print("Accuracy : ", acc)
                    dump(model, 'data/models/modelPrep-'+str(layer1)+'-'+str(layer2)+ '-'+str(layer3)+'.joblib')
                    print("Saving a new model({}, {})".format(layer1, layer2))
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

