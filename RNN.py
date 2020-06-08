import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import csv



data = np.load('data/rnn/dataUnbalanced.npy')
labels = np.load('data/rnn/labelsUnbalanced.npy')
data = np.reshape(data,(data.shape[0],10, 64))
x_train , x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=1)

print(type(x_test))

print(data.shape)
print(x_train[0].shape)
print(x_train.shape[1:])

model = Sequential()


lstm1 = [30]
lstm2 = [30]
dense1 = [30]
dense2 = [30]

for l1 in lstm1:
    for l2 in lstm2:
        for d1 in dense1:
            for d2 in dense2:
              path = "./data/models/rnn/" + str(l1)+'_'+ str(l2)+'_'+ str(d1)#+'_'+ str(d2)
              os.mkdir(path, 0o777)

              model.add(LSTM(l1, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
              model.add(Dropout(0.1))



              model.add(LSTM(l2, activation='relu'))
              model.add(Dropout(0.05))


              model.add(Dense(d1, activation='relu'))
              model.add(Dropout(0.05))

              model.add(Dense(d2, activation='relu'))
              model.add(Dropout(0.05))

              model.add(Dense(3, activation='softmax'))

              opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

              # Compile model
              model.compile(
                  loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'],
              )

              checkpoint_filepath = path
              model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                  filepath=checkpoint_filepath,
                  save_weights_only=False,
                  monitor='val_accuracy',
                  mode='max',
                  save_best_only=True)

              model.fit(x_train,
                        y_train,
                        epochs=50,
                      #   batch_size=16,
                        callbacks=[model_checkpoint_callback],
                        validation_data=(x_test, y_test))
              
              loss, acc = model.evaluate(x_test,  y_test, verbose=2)
              print(f"final accuracy {acc}")
              
              with open('data/models/rnn/modelResults.csv', mode='a') as csv_file:
                fieldnames = ['Name', 'Acc']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'Name': path, 'Acc':acc})


