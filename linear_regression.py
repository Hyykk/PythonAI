import numpy as np
import matplotlib.pyplot as plt
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])
a = 0
b = 0
lr = 0.03 # learning rate
epochs = 2001# epochs
n = len(x)# number of x
# gradient decent
for i in range(epochs):
    y_pred = a * x + b
    error = y - y_pred

    a_diff = (2/n) * sum(-x * error)
    b_diff = (2/n) * sum(-error)
    a = a - lr * a_diff
    b = b - lr * b_diff
    if(i % 100 == 0):
        print('epoch: {}, a: {}, b: {}'.format(i, a, b))

plt.scatter(x, y)
plt.plot(x, y_pred, 'r')
plt.show()
