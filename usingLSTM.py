from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt
#data preprocessing
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words = 1000, test_split = 0.2)
#print(type((X_train, y_train)))
#category = np.max(y_train) + 1
#print('# of Category: ',category)
#print('# of news for training: ', len(X_train))
#print('# of news for testing: ', len(X_test))
#print(X_train[0])
X_train = sequence.pad_sequences(X_train, 100)
X_test = sequence.pad_sequences(X_test, 100)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# model structure
model = Sequential()
model.add(Embedding(1000, 100))
model.add(LSTM(100, activation = 'tanh'))
model.add(Dense(46, activation = 'softmax'))
# model train
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 5)
history = model.fit(X_train, y_train, batch_size = 20, epochs = 200, validation_data = (X_test, y_test), callbacks = [early_stopping_callback])
print('\n Test Accuracy:', model.evaluate(X_test, y_test)[1])
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
