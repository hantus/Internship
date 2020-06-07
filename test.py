import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical






new_model = tf.keras.models.load_model('data/models/rnn/30_30_30')

data = np.load('data/rnn/dataUnbalanced.npy')
labels = np.load('data/rnn/labelsUnbalanced.npy')
data = np.reshape(data,(data.shape[0],10, 64))
x_train , x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=1)

new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(x_test,  y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

# print(new_model.predict(x_test).shape)
# pred = new_model.predict(x_test)
# for i in range(10):
    
#     print(np.argmax(pred[i]))


# mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
# (x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

# x_train = x_train/255.0
# x_test = x_test/255.0

# data = np.load('data/rnn/data.npy')
# labels = np.load('data/rnn/labels.npy')
# data = np.reshape(data,(data.shape[0],10, 64))
# x_train , x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=1)

# print(type(x_test))

# print(data.shape)
# print(x_train[0].shape)
# print(x_train.shape[1:])

# model = Sequential()

# # IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
# model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
# model.add(Dropout(0.1))

# model.add(LSTM(128, activation='relu'))
# model.add(Dropout(0.05))

# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.05))

# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.05))

# model.add(Dense(3, activation='softmax'))

# opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# # Compile model
# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=opt,
#     metrics=['accuracy'],
# )

# model.fit(x_train,
#           y_train,
#           epochs=30,
#           validation_data=(x_test, y_test))

# # preds = model.predict(x_test, y_test)
# # print(preds)
