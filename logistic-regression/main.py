import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(s):
    return 1.0 / (1 + np.exp(-s))


def prob(w, X):
    return sigmoid(X.dot(w))


def loss(w, X, y, lamda):
    z = prob(w, X)

    return -np.mean(y * np.log(z) + (1 - y) * np.log(1 - z) + 0.5 * (lamda / X.shape[0]) * np.sum(w * w))


def logistic_regression(w_init, X, y, lamda=0.001, eta=.1, nepoches=2000):
    N, d = X.shape[0], X.shape[1]
    w = w_old = w_init
    ep = 0
    loss_his = [loss(w, X, y, lamda)]
    while ep < nepoches:
        ep += 1
        idx_shuffle = np.random.permutation(N)
        for i in idx_shuffle:
            xi = X[i]
            yi = y[i]

            zi = sigmoid(xi.dot(w))
            w = w - eta * ((zi - yi) * xi + lamda * w)
        loss_his.append(loss(w, X, y, lamda))
        if np.linalg.norm(w - w_old) / d < 1e-6:
            break
        w_old = w
    return w, loss_his


X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
               2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
# bias trick
N = 20
for i in range(20):
    if (y[i] == 0):
        plt.plot(X.T[0][i], y[i], 'o')
    else:
        plt.plot(X.T[0][i], y[i], 'x')


Xbar = np.concatenate((X, np.ones((N, 1))), axis=1)
w_init = np.random.randn(Xbar.shape[1])
lam = 0.0001
w, loss_hist = logistic_regression(w_init, Xbar, y, lam, eta=0.05, nepoches=500)

x_f=[]
for i in range(20):
   x_f.append(10-i)
y_f = []
for xi in x_f:
    y_f.append(sigmoid(xi))
plt.plot(x_f, y_f)
plt.show()

print(loss(w, Xbar, y, lam))
