import numpy as np

# L(w) = sigma i: 1 -> N [( a * x_i + b ) - y_i]
# L(w) min -> gradient L'(w) = 0

# X = [[x_1,1],
#      [x_2,1],
#      ...
#      [x_N,1]]

# w = [a,b]

# y = [y_1, y_2, ...,y_N]

# L(w) = ||Xw-y||2^2
# L'(w) = X^T*(Xw-y)
# L'(w) = 0 <=> X^T*X*w = X^T*y
#           <=> w = (X^T*X)^-1*X^T*y

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

ones = np.array([np.ones(X.shape[0])]).T
Xbar = np.concatenate((X,ones),axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)

# get result [a,b]
w_result = np.linalg.pinv(A).dot(b)

print(w_result)