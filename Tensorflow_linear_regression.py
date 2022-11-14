import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

model = Sequential()
model.add(Dense(1, input_dim = 1, activation = 'linear'))

model.compile(optimizer = 'sgd', loss = 'mse')
model.fit(x, y, epochs = 500)

y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred, 'r')
plt.show()

#hours = int(input("Input hours as integer))
hours = 7
tutors = 4
exp_score = model.predict([hours, tutor])
print('The expected score is', exp_score)
