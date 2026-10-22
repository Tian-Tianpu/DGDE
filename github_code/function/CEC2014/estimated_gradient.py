import numpy as np


def est_gradient(func, x, nx, Os, Mr, s_flag, r_flag, scale):
    h = 1e-6
    return precise_central_diff(func, x, nx, Os, Mr, s_flag, r_flag, scale, h)

def precise_central_diff(func, x, nx, Os, Mr, s_flag, r_flag, scale, h):
    grad = np.zeros(nx)
    for i in range(nx):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (func(x_plus, nx, Os, Mr, s_flag, r_flag, scale) -
                   func(x_minus, nx, Os, Mr, s_flag, r_flag, scale)) / (2 * h)
    return grad

# def stochastic_coordinate_gradient(func, x, nx, Os, Mr, s_flag, r_flag,scale, h):
#     # 采样数量
#     sample_size = nx // 2
#     # 随机选择坐标
#     indices = np.random.choice(nx, size=sample_size, replace=False)
#     grad = np.zeros(nx)
#     for idx in indices:
#         x_plus, x_minus = x.copy(), x.copy()
#         x_plus[idx] += h
#         x_minus[idx] -= h
#         grad[idx] = (func(x_plus, nx, Os, Mr, s_flag, r_flag, scale) -
#                      func(x_minus, nx, Os, Mr, s_flag, r_flag, scale)) / (2 * h)
#     return grad

#============================6-8===========================(x, nx, Os, Mr, S, s_flag, r_flag, scale)
def S_est_gradient(func, x, nx, Os, Mr, s_flag, r_flag, scale):
    n = len(x)
    h = 1e-6
    if n <= 1000:
        # 低维：中心差分
        return S_precise_central_diff(func, x, nx, Os, Mr, s_flag, r_flag, scale, h)
    else:
        # 高维(>100)：使用随机坐标梯度估计
        return S_stochastic_coordinate_gradient(func, x, nx, Os, Mr,s_flag, r_flag, scale, h)

def S_precise_central_diff(func, x, nx, Os, Mr, s_flag, r_flag, scale, h):
    grad = np.zeros(nx)
    for i in range(nx):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (func(x_plus, nx, Os, Mr, s_flag, r_flag, scale) -
                   func(x_minus, nx, Os, Mr, s_flag, r_flag, scale)) / (2 * h)
    return grad

def S_stochastic_coordinate_gradient(func, x, nx, Os, Mr, s_flag, r_flag, scale, h):
    # 采样数量
    sample_size = 50
    # 随机选择坐标
    indices = np.random.choice(nx, size=sample_size, replace=False)
    grad = np.zeros(nx)
    for idx in indices:
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[idx] += h
        x_minus[idx] -= h
        grad[idx] = (func(x_plus, nx, Os, Mr, s_flag, r_flag, scale) -
                     func(x_minus, nx, Os, Mr, s_flag, r_flag, scale)) / (2 * h)
    return grad