import math
import random

import numpy as np


# find minimum f(x) = x^2 + 5sin(x)
# f'(x) = 2x + 5cos(x)
# => Xét thấy nếu x đang đồng biến thì f'(x) > 0 vậy để x tiến về cực tiểu thì x phải lùi về = x - alpha * f'(x)
# => Nếu x đang nghịch biến thì f'(x) < 0 vậy để x tiến về cực tiểu thì x phải tiến về phía trước = x - alpha*f'(x)
# vậy để x tiến về cực tiểu thì nhìn chung x phải di chuyển ngược dấu với đạo hàm
# alpha là hệ số học (learning-rate)

def grad(x):
    return 2 * x + 5 * math.cos(x)


def f(x):
    return x ** 2 + 5 * math.sin(x)


def gd(eta, x_init):
    x_result = x_init
    for i in range(100):
        x_result = x_result - eta * grad(x_result)
        if abs(grad(x_result) < 1e-3):
            break
    return x_result


random.seed(7)
x_init = random.random()
x_result = gd(0.1, x_init)
print(f(x_result))


# ===========================================================
# khai triển taylor ta có
# f'(x) = (f(x+delta) - f(x-delta)) / 2*delta với delta đủ nhỏ 1-e6
# ===========================================================
# Với gradient-descent đơn thuần có thể giá trị chỉ có thể tiến tới được điểm cực tiểu lân cận
# mà không tìm được điểm cực tiểu toàn phần vì vậy ta cần chỉnh sử lại công thức
# Để x tiến về cực tiểu theta_t+1 = theta_t - v_t
# v_t = Yv_t-1 + n*f'(theta_t-1)

def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new)) / len(theta_new) < 1e-3


def gd_monmentum(theta_init, grad, eta, gamma):
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)

    for it in range(100):
        v_new = v_old * gamma + eta * grad(theta[-1])
        theta_new = theta[-1] - v_new
        if has_converged(theta_new, grad):
            break
        theta.append(theta_new)
        v_old = v_new
    return theta


x_result = gd_monmentum(theta_init=x_init, grad = grad, eta=0.1, gamma=0.9)
print(f(x_result))
