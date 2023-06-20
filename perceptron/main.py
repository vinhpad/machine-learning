import numpy as np
import matplotlib.pyplot as plt


# J(w) = sigma i : 1 -> N ( -y * W^T * x_i )
# J(w;xi;yi) = -yi * w^T * xi || SGD
# J'(w) = -yi*xi
# w -> w - J'(w)*eta = w + yi*xi @@

def predict(w, X):
    return np.sign(X.dot(w))


def perceptron(X, y, w_init):
    w = w_init
    while True:
        pre = predict(w, X)

        mis_point = []
        for idx in range(len(pre)):
            if pre[idx] != y[idx]:
                mis_point.append(idx)

        if len(mis_point) == 0:
            break

        idx_random = np.random.choice(mis_point, 1)[0]

        w = w + y[idx_random] * X[idx_random]

    return w


means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
for i in X0:
    plt.plot(i[0], i[1], 'o')

X1 = np.random.multivariate_normal(means[1], cov, N)
for i in X1:
    plt.plot(i[0], i[1], 'x')

X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -1 * np.ones(N)))
Xbar = np.concatenate((X, np.ones((2 * N, 1))), axis=1)
w_init = np.random.randn(Xbar.shape[1])
w = perceptron(Xbar, y, w_init)

plt.plot([-1,1],[(-w[2] - (-1) * w[0]) / w[1], (-w[2] - 1 * w[0]) / w[1]])

plt.show()
