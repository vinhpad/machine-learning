from __future__ import print_function
import numpy as np
from time import time  # for comparing runing time

d, N = 1000, 10000  # dimension, number of training points
X = np.random.randn(N, d)  # N d-dimensional points
z = np.random.randn(d)
# X.shape = [N,d]
# z.shape = [d]
# distance of point z to point x
def dist_pp(z, x):
    d = z - x.reshape(z.shape)
    return np.sum(d * d)
# distance of point z to [x_i] (slowly)
def dist_ps_naive(z, X):
    N = X.shape[0]
    res = np.zeros((1, N))
    for i in range(N):
        res[0][i] = dist_pp(z, X[i])
    return res

# z = [z_1, z_2, ..., z_d]
# X = [
#   [x_1_1, x_1_2, ..., x_1_d]
#   ...
#   [x_N_1, x_N_2, ..., x_N_d]
# ]
# ||X_i-z||2^2 = (z-X_i)^T * (z-X_i) = ||z||2^2 + ||X_i||2^2 - 2*X_i*z

# from the point to each point in a set, fast
def dist_ps_fast(z, X):
    X2 = np.sum(X * X, 1)
    z2 = np.sum(z*z)
    return X2 + z2 - 2*X.dot(z)

from sklearn import  neighbors,datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(7)
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

X_train, X_test, Y_train, Y_test  = train_test_split(iris_X,iris_Y,test_size=130)

def myweight(distances):
    sigma2 = .4 # we can change this number
    return np.exp(-distances**2/sigma2)

model = neighbors.KNeighborsClassifier(n_neighbors = 5, p = 2, weights= myweight)
model.fit(X_train,Y_train)



y_pred  = model.predict(X_test)
print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(Y_test, y_pred)))
