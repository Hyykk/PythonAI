from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Activation, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import matplotlib.pyplot as plt
# data preprocessing
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 5000)
X_train = pad_sequences(X_train, 500)
X_test = pad_sequences(X_test, 500)
# model structure
model = Sequential()
model.add(Embedding(5000, 100))
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, padding = 'valid', activation = 'relu', strides = 1))
model.add(MaxPooling1D(4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
# model train
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 3)
history = model.fit(X_train, y_train, batch_size = 40, epochs = 100, validation_split = 0.25, callbacks = [early_stopping_callback])
print('\n Test Accuracy by Choi: ', model.evaluate(X_test, y_test)[1])
y_vloss = history.history['val_loss']
y_loss = history.history['loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = '.', color = 'red', label = 'Validation Loss')
plt.plot(x_len, y_loss, marker = '.', color = 'blue', label = 'Train Loss')
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
